from pathlib import Path

import pandas as pd


MANIFEST = Path(os.environ.get("DPRCN_PATCH_MANIFEST", "data/Nanyang_Stage2A_DeterministicPatchManifest.csv"))


def main() -> None:
    table = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if not table.groupby(["date", "stored_DOY"])["split"].nunique().eq(1).all():
        raise AssertionError("同一日期出现在多个split中。")
    sets = {name: set(table.loc[table["split"] == name, "stored_DOY"].unique()) for name in ["train", "val", "test"]}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("train/val/test存在DOY交集。")
    print("PASS: split isolation")


if __name__ == "__main__":
    main()

