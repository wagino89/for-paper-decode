# A Reproducible, Explainable Machine Learning Benchmark for Online Shoppers' Purchasing Intention

Code to reproduce the experiments in:

> **A Reproducible, Explainable Machine Learning Benchmark for Online Shoppers' Purchasing Intention**
> Wagino, Muhammad Subhan, Silvia Ratna, M. Muflih
> Universitas Islam Kalimantan Muhammad Arsyad Al Banjari, Banjarmasin, Indonesia
> *DECODE: Jurnal Pendidikan Teknologi Informasi* (to appear)

This repository provides an **honest, leakage-controlled** benchmark for predicting
online shoppers' purchasing intention on the UCI Online Shoppers dataset, together
with **RAID (Resampling-Aware Inflation Decomposition)** — a diagnostic that explains
why many previously reported, near-perfect scores on this dataset are inflated.

---

## Why this repository

Reported performance on the UCI Online Shoppers dataset varies enormously across
studies. Much of that variation comes from the **evaluation protocol**, not the
model: when resampling (e.g. SMOTE) is applied *before* the train/test split, scores
are measured on an artificially balanced, contaminated test set and look far better
than they are. This project:

1. Builds a strict, anti-leakage pipeline (preprocessing and resampling fit on the
   training partition only).
2. Compares Logistic Regression, Random Forest, XGBoost, and CatBoost under identical
   honest conditions.
3. Adds structural ablation, SHAP explainability, nested cross-validation, threshold
   and probability calibration, and bootstrap confidence intervals.
4. Introduces **RAID**, which measures the inflation caused by resample-before-split
   evaluation and decomposes it into a *test-distribution artefact* and a
   *synthetic-contamination* component.

## Headline results (honest hold-out, seed 42)

| Model                         | Accuracy | Precision | Recall | F1 (minority) | ROC-AUC | PR-AUC |
|-------------------------------|:--------:|:---------:|:------:|:-------------:|:-------:|:------:|
| **XGBoost + feature selection** | **0.8779** | 0.5815 | 0.7565 | **0.6576** | **0.9223** | 0.7257 |
| XGBoost (baseline)            | -        | -         | -      | 0.6629        | -       | -      |
| CatBoost                      | -        | -         | -      | 0.6391        | -       | -      |
| Logistic Regression           | -        | -         | -      | 0.5917        | -       | -      |
| Random Forest                 | -        | -         | -      | 0.5867        | -       | -      |

Weighted-F1 of the final model is **0.8842** (reported so results stay comparable
across the differing protocols used in prior work).

### RAID inflation audit (same model, same preprocessing)

| Arm | Description | F1 | ROC-AUC |
|-----|-------------|:--:|:-------:|
| A   | Leaky: resample before split, balanced test | 0.9392 | 0.9868 |
| A'  | Leaky training, natural test                | 0.6749 | 0.9372 |
| B   | **Honest: SMOTE in training, natural test** | 0.6546 | 0.9306 |

For F1, the total A-B inflation of **+0.2846** decomposes into a
**test-distribution artefact of +0.2643 (~93%)** and **synthetic contamination of
+0.0203 (~7%)** — i.e. the near-perfect numbers are overwhelmingly an artefact of the
balanced test set, not genuine skill.

> Exact values require the pinned environment and `seed = 42` (see
> `requirements.txt`). Small deviations across library versions are expected.

---

## Repository structure

```
for-paper-decode/
├── reproduce_experiments.ipynb   # SELF-CONTAINED notebook: Run All to reproduce everything
├── run_experiments.py      # same pipeline as a CLI (writes CSVs to Results/)
├── src/
│   ├── config.py           # paths, seed, feature lists, hyper-parameters
│   ├── data.py             # loading, split, leakage-safe preprocessors
│   ├── models.py           # model builders (LR, RF, XGBoost, tuned XGBoost)
│   ├── metrics.py          # metrics, F1 variants, bootstrap CI, calibration
│   ├── experiments.py      # comparative, ablation, k-sensitivity, SHAP,
│   │                       #   nested CV, threshold tuning
│   └── raid.py             # RAID protocol (arms A, A', B + decomposition)
├── Dataset/                # put the dataset CSV here (see Dataset/README.md)
├── Results/                # generated CSV tables
│   └── figures/            # generated SHAP figures
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

## Installation

```bash
git clone https://github.com/wagino89/for-paper-decode.git
cd for-paper-decode
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Then download the dataset into `data/` as described in [`Dataset/README.md`](Dataset/README.md).

## Usage

### Option A - one notebook (recommended for reviewers)

Open **`reproduce_experiments.ipynb`** (Jupyter, VS Code, or Google Colab), put the
dataset CSV in `Dataset/`, and choose *Run All*. Every table and figure is produced
top to bottom and written to `Results/`. All code is inline - no package imports from
this repository are required.

### Option B - command line

Run everything:

```bash
python run_experiments.py --all
```

Or run specific stages:

```bash
python run_experiments.py --compare --ablation        # Tables 6 & 7
python run_experiments.py --shap                       # SHAP importance + figures
python run_experiments.py --raid                       # RAID audit (Tables 13-14)
python run_experiments.py --tune                       # re-run XGBoost hyper-parameter search
```

| Flag | Produces |
|------|----------|
| `--compare`     | `Results/table6_comparative.csv` |
| `--ablation`    | `Results/table7_ablation.csv` |
| `--ksens`       | `Results/table8_k_sensitivity.csv` |
| `--shap`        | `Results/table9_shap_importance.csv`, `Results/figures/shap_*.png` |
| `--nestedcv`    | `Results/table10_nested_cv_*.csv` |
| `--threshold`   | `Results/table11_threshold.csv` |
| `--calibration` | `Results/calibration_*.csv` |
| `--bootstrap`   | `Results/table12_bootstrap_ci.csv` |
| `--raid`        | `Results/table13_raid_arms.csv`, `Results/table14_raid_decomposition.csv` |

`--all` runs every stage. See `python run_experiments.py -h` for details.

> The loader reads `Dataset/online_shoppers_intention.csv` by default; if your file
> has a different name it automatically picks the first `*.csv` in `Dataset/`.

## Reproducibility

- Results are written to `Results/` (tables) and `Results/figures/` (SHAP plots).
- A single random seed (`42`) controls the split, resampling, tuning, SHAP sampling,
  and bootstrap.
- All preprocessing and resampling are fit on the training partition only (inside
  scikit-learn / imbalanced-learn pipelines) to avoid leakage.
- The tuned XGBoost hyper-parameters are stored in `src/config.py` and were obtained
  with the `--tune` search (30 iterations, F1 scoring, 5-fold stratified CV).
- Exact library versions are pinned in `requirements.txt`.

## Citation

```bibtex
@article{wagino2026raid,
  title   = {A Reproducible, Explainable Machine Learning Benchmark for
             Online Shoppers' Purchasing Intention},
  author  = {Wagino and Subhan, Muhammad and Ratna, Silvia and Muflih, M.},
  journal = {DECODE: Jurnal Pendidikan Teknologi Informasi},
  year    = {2026},
  note    = {To appear}
}
```

## License

Released under the [MIT License](LICENSE).
