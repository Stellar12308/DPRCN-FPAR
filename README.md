# DPRCN-A3 for fine-resolution FPAR reconstruction

Official code accompanying the manuscript **“Fine-Resolution FPAR Reconstruction Using a Historical Prior-Guided Residual Correction Network”** (manuscript TGRS-2026-04953).

DPRCN-A3 reconstructs 30 m target-date FPAR from four active inputs: a historical Landsat FPAR prior (HP), the log-transformed historical observation count (HN), historical FPAR standard deviation (HU), and contemporaneous MODIS FPAR (MF). The network predicts a bounded correction in logit space and combines it with the historical prior.

## Repository contents

- `dprcn/`: model, masked loss, datasets, and reproducibility utilities.
- `scripts/`: deterministic patch generation, normalization, ablation, multi-seed, final training, held-out testing, and leave-one-season-out evaluation.
- `configs/`: frozen JSON configurations used by the experiments.
- `baselines/`: ESTARFM and adapted PK-STEM comparison code and common-mask evaluation.
- `tests/`: model and loss checks. Data-dependent tests require the processed dataset.

## Environment

The reported experiments used Python 3.10 and PyTorch. Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version from the official PyTorch instructions.

## Data layout and paths

The scripts read `project_root` from their JSON configuration. The public configurations use `.` so commands should be run from the root of a local project containing the processed-data and results directories referenced by each configuration. Alternatively, replace `project_root` with an absolute local path. No raw or processed satellite imagery is included in this repository.

The main experimental sequence is:

```text
Stage2A  deterministic patch manifest
Stage2B  training-only normalization
Stage2C  dataset audit
Stage2D  candidate training
Stage2E  input ablation
Stage2F  multi-seed robustness
Stage2G  final A3 fit on training and validation dates
Stage2H  one-time held-out test evaluation
Stage2I  leave-one-season-out evaluation
Stage3   baseline and common-mask comparisons
```

Run scripts from the repository root, for example:

```bash
python scripts/Nanyang_Stage2G_TrainFinalA3_TrainVal32Epochs.py
python scripts/Nanyang_Stage2H_EvaluateFinalA3_OneTimeTest.py
```

## Reproducibility notes

- The frozen date split contains 16 training, five validation, and five held-out test dates.
- Model selection uses validation data only. The held-out test set is accessed after model development.
- The final DPRCN-A3 model uses seed 42 and 32 fixed training epochs on the combined training and validation sets.
- Evaluation uses valid-pixel masks; NoData pixels are not converted to valid zero FPAR values.
- Configuration files retain the experiment names used in the manuscript and supplementary material.

## Data availability

Processed data supporting the study are available from the corresponding author upon reasonable request, subject to the distribution conditions of the source satellite products. Original Landsat, MODIS, and SRTM products are available from their respective public data providers.

## Citation

Please cite the associated manuscript if you use this code. Full bibliographic details will be added after publication.
