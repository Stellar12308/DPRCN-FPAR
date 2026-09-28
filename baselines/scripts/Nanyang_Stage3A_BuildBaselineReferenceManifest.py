from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve()
CFG = HERE.parents[1] / "configs" / "Nanyang_Stage3A_BaselineReferenceManifest_Config_v1.json"
with CFG.open("r", encoding="utf-8") as file:
    cfg = json.load(file)
root = Path(cfg["project_root"])
source = root / "04_Results/Stage2A_PatchManifest/Nanyang_Stage2A_DeterministicPatchManifest.csv"
patches = pd.read_csv(source, encoding="utf-8-sig", dtype={"date": str})
patches["date"] = patches["date"].astype(str).str.zfill(8)
scenes = patches.sort_values("patch_id").drop_duplicates("date").copy()
scenes["calendar_date"] = pd.to_datetime(scenes["date"], format="%Y%m%d")
pool = scenes.loc[scenes["split"].isin(cfg["reference_pool_splits"])]
targets = scenes.loc[scenes["split"].eq(cfg["target_split"])]
if len(pool) != 21 or len(targets) != 5:
    raise ValueError(f"角色数量错误：pool={len(pool)}, targets={len(targets)}")

records = []
for _, target in targets.sort_values("calendar_date").iterrows():
    previous = pool.loc[pool["calendar_date"] < target["calendar_date"]].sort_values("calendar_date").tail(1).iloc[0]
    following = pool.loc[pool["calendar_date"] > target["calendar_date"]].sort_values("calendar_date").head(1).iloc[0]
    if previous["split"] == "test" or following["split"] == "test":
        raise RuntimeError("测试标签进入参考池。")
    records.append({
        "target_date": target["date"], "target_DOY": int(target["stored_DOY"]),
        "target_label_evaluation_only": target["label_path"], "target_modis_input": target["modis_path"],
        "previous_date": previous["date"], "previous_DOY": int(previous["stored_DOY"]),
        "previous_original_split": previous["split"], "previous_fine_label_input": previous["label_path"],
        "previous_modis_input": previous["modis_path"],
        "previous_gap_days": int((target["calendar_date"] - previous["calendar_date"]).days),
        "next_date": following["date"], "next_DOY": int(following["stored_DOY"]),
        "next_original_split": following["split"], "next_fine_label_input": following["label_path"],
        "next_modis_input": following["modis_path"],
        "next_gap_days": int((following["calendar_date"] - target["calendar_date"]).days),
        "test_label_used_as_input": 0, "status": "PASS",
    })

result = pd.DataFrame(records)
out_dir = root / cfg["output_dir"]
out_dir.mkdir(parents=True, exist_ok=True)
result.to_csv(out_dir / "Nanyang_Stage3A_BaselineReferenceManifest_5TestDates.csv", index=False, encoding="utf-8-sig")
print(result[["target_date", "previous_date", "previous_gap_days", "next_date", "next_gap_days"]].to_string(index=False))
print("STAGE3A=PASS")

