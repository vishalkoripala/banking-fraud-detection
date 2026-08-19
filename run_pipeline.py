"""
run_pipeline.py

End-to-end runner for the banking fraud detection pipeline.

Usage:
    python run_pipeline.py

Expects data/creditcard.csv with columns: Time, V1..V28, Amount, Class
(the standard ULB / ML-G Credit Card Fraud Detection schema). Swap in your
own copy of that CSV at the same path and everything below still works
unmodified.
"""
import os
import time

import numpy as np
import pandas as pd

from pipeline import build_model_pipelines
from evaluate import (
    cross_validated_scores, summarize_model, best_f1_threshold,
    plot_pr_curves, plot_roc_curves, plot_confusion_matrix,
)

DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "outputs"
N_SPLITS = 5


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH} ...")
    X, y = load_data(DATA_PATH)
    fraud_rate = y.mean()
    print(f"Rows: {len(X):,} | Features: {X.shape[1]} | Fraud cases: {y.sum():,} ({fraud_rate:.4%})\n")

    pipelines = build_model_pipelines()
    oof_probas = {}
    summaries = []

    for name, pipe in pipelines.items():
        print(f"Running {N_SPLITS}-fold stratified CV: {name} ...")
        t0 = time.time()
        probas = cross_validated_scores(pipe, X, y, n_splits=N_SPLITS)
        elapsed = time.time() - t0
        oof_probas[name] = probas

        thresh, f1 = best_f1_threshold(y.values, probas)
        summary = summarize_model(name, y.values, probas, threshold=thresh)
        summaries.append(summary)

        print(f"  done in {elapsed:.1f}s")
        print(f"  Average Precision (PR-AUC): {summary['average_precision']:.4f}")
        print(f"  ROC-AUC:                    {summary['roc_auc']:.4f}")
        print(f"  Best-F1 threshold:          {thresh:.6f}  (F1={f1:.4f})")
        print(f"  (for reference) precision/recall at the default 0.5 cutoff:")
        default_pred = (probas >= 0.5).astype(int)
        from sklearn.metrics import precision_score, recall_score
        print(f"    precision={precision_score(y.values, default_pred):.4f}  "
              f"recall={recall_score(y.values, default_pred):.4f}  "
              f"(class_weight='balanced' skews raw probabilities -- see README)")
        print(summary["report"])

        plot_confusion_matrix(
            summary["confusion_matrix"],
            f"{OUTPUT_DIR}/confusion_matrix_{name}.png",
            title=f"Confusion Matrix — {name}\n(threshold={thresh:.3f}, chosen to maximize F1)",
        )
        print(f"  saved outputs/confusion_matrix_{name}.png\n")

    plot_pr_curves(oof_probas, y.values, f"{OUTPUT_DIR}/pr_curve_comparison.png")
    plot_roc_curves(oof_probas, y.values, f"{OUTPUT_DIR}/roc_curve_comparison.png")
    print(f"Saved outputs/pr_curve_comparison.png")
    print(f"Saved outputs/roc_curve_comparison.png\n")

    print(f"Naive 'flag nothing' accuracy would be {(1 - fraud_rate):.4%} "
          f"while catching 0% of fraud -- why this project scores on PR-AUC, not accuracy.\n")

    with open(f"{OUTPUT_DIR}/metrics_report.md", "w") as f:
        f.write("# Fraud Detection Pipeline — Metrics Report\n\n")
        f.write(f"- Dataset: {len(X):,} transactions, {y.sum():,} confirmed fraud ({fraud_rate:.4%})\n")
        f.write(f"- Validation: {N_SPLITS}-fold StratifiedKFold, out-of-fold predictions "
                f"(cross_val_predict) so every metric below is on held-out data\n\n")
        f.write(f"Naive majority-class baseline: {(1 - fraud_rate):.4%} accuracy, "
                f"0% recall, undefined precision -- included to show why accuracy is the "
                f"wrong metric for this problem.\n\n")
        f.write(
            "**Note on thresholds:** both models use `class_weight=\"balanced\"` to cope "
            "with the 0.17% fraud rate. That reweights the training loss as if classes were "
            "roughly 50/50, which pushes predicted probabilities to reflect that artificial "
            "balance rather than the true prior -- so the conventional 0.5 cutoff is not a "
            "sensible operating point here (for logistic regression it yields ~6% precision "
            "at ~91% recall). Each threshold below was instead chosen from the out-of-fold "
            "PR curve to maximize F1, which is the whole reason this project builds a PR "
            "curve rather than reading off one accuracy number.\n\n"
        )

        for s in summaries:
            f.write(f"## {s['name']}\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Average Precision (PR-AUC) | {s['average_precision']:.4f} |\n")
            f.write(f"| ROC-AUC | {s['roc_auc']:.4f} |\n")
            f.write(f"| Best-F1 threshold | {s['threshold']:.6f} |\n")
            f.write(f"| F1 at that threshold | {s['f1_at_threshold']:.4f} |\n\n")
            f.write("```\n" + s["report"] + "\n```\n\n")

    print(f"Saved {OUTPUT_DIR}/metrics_report.md")


if __name__ == "__main__":
    main()
