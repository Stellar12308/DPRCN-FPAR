from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


TEST_PATH = Path(__file__).resolve()
STAGE2_ROOT = TEST_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.dataset import NanyangFPARPatchDataset, build_dataloader  # noqa: E402
from dprcn.losses import build_loss  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    config = load_json(
        STAGE2_ROOT / "configs" / "Nanyang_Stage2D_Training_Config_v1.json"
    )
    seed_everything(int(config["random_seed"]))
    root = Path(config["project_root"])
    patch_path = root / config["stage2a_patch_manifest"]
    normalization_path = root / config["stage2b_parameters"]
    normalization = load_json(normalization_path)
    dataset = NanyangFPARPatchDataset(
        patch_manifest_path=patch_path,
        normalization_path=normalization_path,
        split="train",
        scene_cache_size=2,
    )
    loader = build_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=int(config["random_seed"]),
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, normalization).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    inputs = batch["input"].to(device)
    target = batch["target"].to(device)
    mask = batch["valid_mask"].to(device)
    weight = batch["weight"].to(device)
    losses = []
    model.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)["prediction"]
        loss = criterion(prediction, target, mask, weight)["total"]
        if not torch.isfinite(loss):
            raise RuntimeError("真实Batch过拟合出现非有限损失。")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    reduction = (losses[0] - losses[-1]) / losses[0]
    if not losses[-1] < losses[0] or reduction < 0.20:
        raise RuntimeError(
            f"真实Batch损失下降不足：{losses[0]:.6f}->{losses[-1]:.6f}"
        )
    print("REAL_BATCH_OVERFIT_TEST=PASS")
    print(f"loss_initial={losses[0]:.8f}")
    print(f"loss_final={losses[-1]:.8f}")
    print(f"relative_reduction={reduction:.6f}")
    print("test_split_accessed=0")


if __name__ == "__main__":
    main()

