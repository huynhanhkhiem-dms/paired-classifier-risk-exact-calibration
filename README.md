# Exact Calibration for Paired Classifier Risk on Finite Evaluation Pools

Research code accompanying the manuscript **"Exact Calibration for Paired Classifier Risk on Finite Evaluation Pools"**.

**Author:** Huynh Anh Khiem · Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam  
**ORCID:** https://orcid.org/0009-0007-7210-174X

## Requirements

Python 3.12. Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduction workflow

Run from the repository root. First regenerate prediction pools from the UCI source datasets:

```bash
python prepare_pools.py --raw downloaded --output pools
```

Then run the exact-verification and experiments:

```bash
python verify_exact.py --output reproduced/verification.json
python exact_power.py --output reproduced
python run_experiments.py --output reproduced/audit_trials.csv
python panel_experiment.py --output reproduced/matched_panels.csv
python calibration_time.py --output reproduced/calibration_times.csv
python summarize.py --results reproduced --output reproduced/manuscript_assets
```

The `paired_exact.py` and `recalibration.py` modules implement exact tests using integer counts and rational thresholds. `prepare_pools.py` checks downloaded UCI source archives against `source_checksums.csv`. The original experiment used the random seed **20261116**. Timing numbers are hardware-dependent; retraining model predictions can differ across library environments.

The accompanying journal supplementary package additionally contains the fixed prediction pools and experiment output files. This repository's source-code reproduction pathway can regenerate them; do not treat newly regenerated outputs as bitwise identical to the archived results without checking.

## Dataset attribution

Six datasets from the UCI Machine Learning Repository, provided under CC BY 4.0:

- Optical Recognition of Handwritten Digits — https://doi.org/10.24432/C50P49
- Pen-Based Recognition of Handwritten Digits — https://doi.org/10.24432/C5MG6K
- Statlog (Landsat Satellite) — https://doi.org/10.24432/C55887
- Letter Recognition — https://doi.org/10.24432/C5ZP40
- Dry Bean — https://doi.org/10.24432/C50S4B
- Human Activity Recognition Using Smartphones — https://doi.org/10.24432/C54S4K

Data attribution and methodological definitions are documented in the supplementary README. Code licensing is not asserted until the rights holder selects a license.
