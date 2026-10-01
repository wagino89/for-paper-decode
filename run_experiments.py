#!/usr/bin/env python3
"""
Reproduce the experiments of:

    "A Reproducible, Explainable Machine Learning Benchmark for
     Online Shoppers' Purchasing Intention"

Usage
-----
    python run_experiments.py --all
    python run_experiments.py --compare --ablation --raid
    python run_experiments.py --tune            # regenerate tuned XGBoost params

All result tables are written as CSV to results/ and SHAP figures to figures/.
Run `python run_experiments.py -h` for the full list of stages.
"""
import argparse

import pandas as pd

from src import config
from src.data import load_dataset, split_xy, train_test
from src.experiments import (ablation, comparative, k_sensitivity, nested_cv,
                             shap_importance, threshold_and_probs, tune_xgb)
from src.metrics import bootstrap_ci, calibration, f1_variants
from src.raid import run_raid


def _save(df, name):
    config.ensure_dirs()
    path = config.RESULTS_DIR / name
    df.to_csv(path, index=False)
    print(f"  saved -> {path.relative_to(config.ROOT_DIR)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATASET_PATH), help="path to the dataset CSV")
    ap.add_argument("--all", action="store_true", help="run every stage")
    ap.add_argument("--tune", action="store_true", help="re-run XGBoost hyper-parameter search")
    ap.add_argument("--compare", action="store_true", help="4-model comparison (Table 6)")
    ap.add_argument("--ablation", action="store_true", help="structural ablation (Table 7)")
    ap.add_argument("--ksens", action="store_true", help="feature-count k-sensitivity (Table 8)")
    ap.add_argument("--shap", action="store_true", help="SHAP global importance + figures")
    ap.add_argument("--nestedcv", action="store_true", help="nested cross-validation (Table 10)")
    ap.add_argument("--threshold", action="store_true", help="threshold tuning (Table 11)")
    ap.add_argument("--calibration", action="store_true", help="Brier score + reliability")
    ap.add_argument("--bootstrap", action="store_true", help="bootstrap 95% CIs (Table 12)")
    ap.add_argument("--raid", action="store_true", help="RAID inflation audit (Tables 13-14)")
    args = ap.parse_args()

    stages = ["tune", "compare", "ablation", "ksens", "shap",
              "nestedcv", "threshold", "calibration", "bootstrap", "raid"]
    if args.all:
        for s in stages:
            setattr(args, s, True)
    if not any(getattr(args, s) for s in stages):
        ap.error("choose at least one stage (e.g. --all, or --compare --raid)")

    print("Loading dataset ...")
    df = load_dataset(args.data)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test(X, y)
    print(f"  train={X_train.shape}  test={X_test.shape}  positives={int(y.sum())}/{len(y)}")

    # cache test-set probabilities for calibration / bootstrap
    test_prob = best_t = None

    if args.tune:
        print("\n[tune] RandomizedSearchCV for XGBoost ...")
        best = tune_xgb(X_train, y_train)
        _save(pd.DataFrame([best]), "tuned_xgb_params.csv")

    if args.compare:
        print("\n[compare] 4-model benchmark ...")
        df_c = comparative(X_train, y_train, X_test, y_test)
        print(df_c.to_string(index=False)); _save(df_c, "table6_comparative.csv")

    if args.ablation:
        print("\n[ablation] structural ablation ...")
        df_a = ablation(X_train, y_train, X_test, y_test)
        print(df_a.to_string(index=False)); _save(df_a, "table7_ablation.csv")

    if args.ksens:
        print("\n[ksens] feature-count sensitivity ...")
        df_k = k_sensitivity(X_train, y_train, X_test, y_test)
        print(df_k.to_string(index=False)); _save(df_k, "table8_k_sensitivity.csv")

    if args.shap:
        print("\n[shap] SHAP global importance ...")
        df_s = shap_importance(X_train, y_train, X_test, y_test)
        print(df_s.head(10).to_string(index=False)); _save(df_s, "table9_shap_importance.csv")

    if args.nestedcv:
        print("\n[nestedcv] nested cross-validation ...")
        folds, summary = nested_cv(X, y)
        print(summary.to_string(index=False))
        _save(folds, "table10_nested_cv_folds.csv"); _save(summary, "table10_nested_cv_summary.csv")

    if args.threshold or args.calibration or args.bootstrap:
        print("\n[threshold] validation-based threshold tuning ...")
        df_t, test_prob, best_t = threshold_and_probs(X_train, y_train, X_test, y_test)
        print(df_t.to_string(index=False)); _save(df_t, "table11_threshold.csv")

    if args.calibration:
        print("\n[calibration] Brier score + reliability ...")
        brier, rel = calibration(y_test, test_prob)
        print("Brier score:", brier); print(rel.to_string(index=False))
        _save(rel, "calibration_reliability.csv")
        _save(pd.DataFrame([{"Model": "Tuned XGBoost + FS", "Brier Score": brier}]),
              "calibration_brier.csv")

    if args.bootstrap:
        print("\n[bootstrap] 95% confidence intervals ...")
        ci_default = bootstrap_ci(y_test, test_prob, threshold=0.50)
        ci_default.insert(0, "Setting", "Default Threshold (0.50)")
        ci_sel = bootstrap_ci(y_test, test_prob, threshold=best_t)
        ci_sel.insert(0, "Setting", f"Validation-Selected Threshold ({best_t:.2f})")
        ci = pd.concat([ci_default, ci_sel], ignore_index=True)
        print(ci.to_string(index=False)); _save(ci, "table12_bootstrap_ci.csv")

    if args.raid:
        print("\n[raid] Resampling-Aware Inflation Decomposition ...")
        arms, decomp = run_raid(X, y, X_train, y_train, X_test, y_test)
        print(arms.to_string(index=False)); print(); print(decomp.to_string(index=False))
        _save(arms, "table13_raid_arms.csv"); _save(decomp, "table14_raid_decomposition.csv")

    print("\nDone. Tables in Results/ , figures in Results/figures/.")


if __name__ == "__main__":
    main()
