# Resampling-Aware Inflation Decomposition (RAID): A Leakage-Controlled Benchmark for Online Purchase-Intention Prediction

Code and results for:

> **Resampling-Aware Inflation Decomposition (RAID): A Leakage-Controlled Benchmark for Online Purchase-Intention Prediction**
> Wagino, Muhammad Subhan, Silvia Ratna, M. Muflih
> Universitas Islam Kalimantan Muhammad Arsyad Al Banjari, Banjarmasin, Indonesia
> *DECODE: Jurnal Pendidikan Teknologi Informasi* (revised manuscript, under review)

Reported scores on the UCI Online Shoppers Purchasing Intention dataset range from a minority-class F1 of about 0.60
to above 0.93. This repository provides:

1. **A leakage-controlled benchmark.** Preprocessing, feature selection and resampling are fitted on training data only,
   inside scikit-learn / imbalanced-learn pipelines. Logistic Regression, Random Forest, XGBoost and CatBoost are
   compared on a held-out test set and by repeated 5×5 cross-validation on the training set.
2. **RAID**, a protocol that measures how much a resample-before-split evaluation inflates a score, and splits that
   inflation into three additive parts:

   ```
   A − B = (A − A_real_bal)      synthetic test points
         + (A_real_bal − A_real) prevalence shift (balanced vs natural class ratio)
         + (A_real − B)          training contamination + test sampling
   ```

   Arm A is the leakage-prone protocol (SMOTE on all data, then split). Arm B is the leakage-controlled protocol (SMOTE inside the
   training pipeline). A_real_bal and A_real score A's model on the real rows of its own test fold only, reweighted to
   50/50 or kept at the natural prevalence.

## Main results

All results come from the stratified 80/20 split (seed 42): 9,864 training and 2,466 test sessions, 382 of them purchases.

**Final model (tuned XGBoost + mutual-information feature selection, k = 20), test set**

| Metric | Test set | Bootstrap 95% CI | Nested 5×5 CV (mean ± SD) |
|---|:-:|:-:|:-:|
| F1 (minority class) | **0.6576** | 0.6206–0.6935 | 0.6664 ± 0.0071 |
| ROC-AUC | 0.9223 | 0.9085–0.9346 | 0.9287 ± 0.0040 |
| PR-AUC | 0.7257 | 0.6804–0.7663 | 0.7359 ± 0.0113 |
| Accuracy | 0.8779 | 0.8642–0.8905 | 0.8797 ± 0.0074 |

Weighted F1 = 0.8842, macro F1 = 0.7916. Repeated 5×5 CV ranks XGBoost and CatBoost as statistically tied on F1
(corrected resampled t-test, p = 0.17). Platt recalibration lowers the Brier score from 0.0873 to 0.0737.

**RAID: minority-class F1 inflation, 30 seeds (mean, share of the total)**

| Classifier | A − B | Synthetic test points | Prevalence shift | Contamination + sampling |
|---|:-:|:-:|:-:|:-:|
| XGBoost (tuned) | +0.281 | +0.175 (63%) | +0.090 (32%) | +0.015 (5%) |
| CatBoost | +0.272 | +0.160 (59%) | +0.096 (35%) | +0.017 (6%) |
| Random Forest | +0.300 | +0.173 (58%) | +0.087 (29%) | +0.040 (13%) |
| Logistic Regression | +0.228 | +0.015 (7%) | +0.203 (90%) | +0.009 (4%) |

Most of the inflation comes from how the test set is built, not from genuine predictive skill. For tree ensembles the main
source is the synthetic points in the test fold; for Logistic Regression it is the balanced class ratio.

## Repository structure

```
├── reproduce_experiments.ipynb   # single notebook: Run All reproduces every table and figure
├── requirements.txt              # pinned library versions
├── Dataset/                      # put the UCI CSV here (see Dataset/README.md)
├── Results/                      # CSV outputs, named after the manuscript tables
│   └── figures/                  # Figures 2–5 (Figure 1 is a schematic)
├── LICENSE                       # MIT
└── README.md
```

| File in `Results/` | Manuscript |
|---|---|
| `table06_test_set.csv`, `table06_cv_*.csv` | Table 6: model comparison (test set, repeated 5×5 CV, corrected t-tests) |
| `table07_ablation_*.csv`, `table07_paired_tests.csv` | Table 7: one-factor ablation, paired bootstrap and McNemar tests |
| `table08_shap_grouped.csv`, `figures/figure2_*.png`, `figures/figure3_*.png` | Table 8, Figures 2–3: SHAP |
| `table09_nested_cv_*.csv`, `table10_bootstrap_ci.csv` | Tables 9–10: nested CV and bootstrap CI |
| `figures/figure4_reliability.png`, `S9_calibration.csv` | Figure 4, Supplementary Table S9: recalibration |
| `raid_runs.csv`, `table11_raid_arms.csv`, `table12_raid_decomposition.csv`, `figures/figure5_raid_components.png` | Tables 11–12, Figure 5, Supplementary Tables S1–S3: RAID |
| `S4_*.csv` … `S8_*.csv` | Supplementary Tables S4–S8 |

## How to run

```bash
git clone https://github.com/wagino89/for-paper-decode.git
cd for-paper-decode
pip install -r requirements.txt
# download the dataset into Dataset/ (see Dataset/README.md), then:
jupyter lab reproduce_experiments.ipynb      # Run All
```

The notebook also runs in **Google Colab**: open it, choose *Run all*, let cell 0.1 install the pinned versions and
restart the runtime, then choose *Run all* again. The default Colab paths are set in cell 0.2. A full run takes about
1.5–2.5 hours on a Colab CPU; most of that time is nested CV and RAID. Individual sections can be switched off with the
`RUN` dictionary in cell 0.2. The RAID section saves every finished (seed, classifier) pair, so an interrupted run resumes
where it stopped.

## Reproducibility

- Python 3, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.4.1, catboost 1.2.10, imbalanced-learn 0.14.2,
  shap 0.52.0 (`requirements.txt`). Other versions can change results in the third or fourth decimal.
- Seed 42 controls the split, tuning, feature selection, SHAP sampling and bootstrap; RAID uses seeds 0–29.
- The notebook checks that the dataset has 12,330 rows and that the test set holds 382 purchases before it runs.
- The files in `Results/` were produced by this notebook. The RAID runs (30 seeds × 4 classifiers) and the recalibration
  were run in Google Colab; all other sections were re-run independently and reproduce the manuscript values exactly
  (a re-run of RAID seed 0 for all four classifiers also matched exactly). The isotonic-regression row of Table S9 is
  sensitive to ties in the predicted probabilities and can differ in the third or fourth decimal across hardware
  (ECE 0.0141 in Colab, 0.0108 on another machine); all other values are hardware-independent.
- Tuned XGBoost hyperparameters (RandomizedSearchCV, 30 candidates, F1, stratified 5-fold CV on the training set):
  `n_estimators=617, max_depth=6, learning_rate=0.0232, subsample=0.736, colsample_bytree=0.741, gamma=0.183,
  min_child_weight=1, reg_alpha=0, reg_lambda=1`.

## Changes from the first version of this repository

- RAID now uses the three-term decomposition above. In the earlier two-term version, arm A′ was defined differently
  in the paper and in the code, and in the code version A′'s model had seen 447 of the 2,466 test rows. A′ is still
  reported, with the definition of the submitted paper, so the two versions can be compared.
- The ablation starts from one reference (tuned XGBoost, all features) and changes one factor at a time. The earlier
  "baseline" row was the untuned model.
- New analyses: repeated-CV model comparison with corrected t-tests, paired bootstrap and McNemar tests, SHAP grouped
  by original variable, Platt/isotonic recalibration, CatBoost ablation and a deduplicated re-run.
- `run_experiments.py` and `src/` were removed; the notebook is the single source of all results.

## Data

Sakar, C. O., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5F88Q

## Citation

```bibtex
@article{wagino2026raid,
  title   = {Resampling-Aware Inflation Decomposition ({RAID}): A Leakage-Controlled Benchmark
             for Online Purchase-Intention Prediction},
  author  = {Wagino and Subhan, Muhammad and Ratna, Silvia and Muflih, M.},
  journal = {DECODE: Jurnal Pendidikan Teknologi Informasi},
  year    = {2026},
  note    = {Revised manuscript under review}
}
```

## License

Released under the [MIT License](LICENSE).
