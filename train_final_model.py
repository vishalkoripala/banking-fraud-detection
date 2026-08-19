"""
train_final_model.py

Produces the artifact that actually gets deployed, with an honest,
reproducible performance number attached to it.

Flow
----
1. One stratified 80/20 train/test split, held fixed (random_state=42).
2. Model comparison + threshold selection happens via 5-fold CV on the
   TRAIN split only (reuses evaluate.py) -- the test split is never touched
   during this stage, so it can't leak into model or threshold choice.
3. The winning pipeline is refit on the full train split.
4. It is scored EXACTLY ONCE on the held-out test split. Those numbers,
   not the CV numbers, are what ship in model_metadata.json and the model
   card, because they're the only numbers produced by a model that never
   saw that data in any capacity.
5. Model + preprocessing (as one fitted sklearn Pipeline) + metadata are
   persisted to model/. The deployed artifact is literally the object that
   produced the reported metrics -- nothing is retrained on more data
   after evaluation, so the claimed numbers stay true of what's shipped.

Usage:
    python train_final_model.py
"""
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from pipeline import build_model_pipelines
from evaluate import cross_validated_scores, best_f1_threshold, summarize_model

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "model"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud)  |  "
          f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)\n")

    # --- Model + threshold selection: TRAIN split only ---
    candidates = build_model_pipelines(random_state=RANDOM_STATE)
    best_name, best_ap = None, -1.0
    train_oof_probas = {}

    for name, pipe in candidates.items():
        print(f"CV on train split: {name} ...")
        probas = cross_validated_scores(pipe, X_train, y_train, n_splits=5)
        train_oof_probas[name] = probas
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_train.values, probas)
        print(f"  train-CV PR-AUC: {ap:.4f}")
        if ap > best_ap:
            best_ap, best_name = ap, name

    print(f"\nSelected model: {best_name} (train-CV PR-AUC={best_ap:.4f})")

    threshold, train_f1 = best_f1_threshold(y_train.values, train_oof_probas[best_name])
    print(f"Threshold chosen from train-CV PR curve: {threshold:.6f} (train F1={train_f1:.4f})")

    # --- Refit winner on the FULL train split ---
    final_pipeline = candidates[best_name]
    print(f"\nRefitting {best_name} on the full train split ({len(X_train):,} rows) ...")
    final_pipeline.fit(X_train, y_train)

    # --- Evaluate ONCE on the untouched test split ---
    test_probas = final_pipeline.predict_proba(X_test)[:, 1]
    test_summary = summarize_model(best_name, y_test.values, test_probas, threshold=threshold)

    print("\n=== Held-out test performance (never used in training or tuning) ===")
    print(f"PR-AUC (Average Precision): {test_summary['average_precision']:.4f}")
    print(f"ROC-AUC:                    {test_summary['roc_auc']:.4f}")
    print(f"F1 at deployed threshold:   {test_summary['f1_at_threshold']:.4f}")
    print(test_summary["report"])

    # --- Persist ---
    model_path = f"{MODEL_DIR}/fraud_model.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"Saved model -> {model_path}")

    cm = test_summary["confusion_matrix"]
    metadata = {
        "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "feature_order": feature_order,
        "decision_threshold": float(threshold),
        "threshold_selection_method": "max F1 on 5-fold out-of-fold PR curve, train split only",
        "training_rows": int(len(X_train)),
        "training_fraud_rows": int(y_train.sum()),
        "test_rows": int(len(X_test)),
        "test_fraud_rows": int(y_test.sum()),
        "test_metrics": {
            "pr_auc": float(test_summary["average_precision"]),
            "roc_auc": float(test_summary["roc_auc"]),
            "f1_at_threshold": float(test_summary["f1_at_threshold"]),
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]), "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]), "true_positive": int(cm[1, 1]),
            },
        },
        "note": "test_metrics come from a single evaluation on the 20% holdout the model "
                "never saw during CV, model selection, threshold tuning, or fitting.",
    }
    meta_path = f"{MODEL_DIR}/model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {meta_path}")

    # A small background sample is all the SHAP explainer needs at serving
    # time -- decouples the deployed API from requiring the full 150MB
    # training set just to explain predictions.
    background_path = f"{MODEL_DIR}/background_sample.csv"
    X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE).to_csv(background_path, index=False)
    print(f"Saved explainability background sample -> {background_path}")


if __name__ == "__main__":
    main()
