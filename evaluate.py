"""
evaluate.py

Cross-validated diagnostics for the fraud pipeline, with an emphasis on
Precision-Recall analysis rather than plain accuracy.

Why PR-AUC instead of accuracy for this dataset: with fraud at ~0.17% of
transactions, a model that predicts "legit" for every single row scores
99.83% accuracy while catching zero fraud. Accuracy can't tell that model
apart from a genuinely useful one. Precision and recall can, because they're
computed against the fraud class specifically rather than against "did you
match the label," so a lazy majority-class model scores near zero on both.
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before pyplot import

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, f1_score,
)


def cross_validated_scores(pipeline, X, y, n_splits=5, random_state=42):
    """
    Runs stratified k-fold CV and returns out-of-fold predicted probabilities
    for the positive (fraud) class.

    StratifiedKFold specifically (not plain KFold) matters here: with only
    492 fraud rows total, a random split could easily leave a fold with
    almost no fraud examples to learn from or validate against. Stratified
    splitting keeps the ~0.17% fraud ratio consistent in every fold.

    cross_val_predict guarantees each row's prediction comes from a model
    that never saw that row during training, so the resulting PR/ROC curves
    reflect genuine out-of-sample performance rather than optimistic
    in-sample fit.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probas = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    return probas


def best_f1_threshold(y_true, y_proba):
    """Finds the probability threshold that maximizes F1 on these predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s[:-1])  # last precision/recall point has no threshold
    return thresholds[best_idx], f1s[best_idx]


def summarize_model(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "name": name,
        "average_precision": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1_at_threshold": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=["Legit", "Fraud"], digits=4),
        "threshold": threshold,
    }


def plot_pr_curves(proba_by_model, y_true, save_path):
    plt.figure(figsize=(7, 6))
    for name, y_proba in proba_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")

    baseline = y_true.sum() / len(y_true)
    plt.hlines(baseline, 0, 1, linestyles="--", colors="gray",
               label=f"No-skill baseline ({baseline:.4f})")
    plt.xlabel("Recall (share of fraud caught)")
    plt.ylabel("Precision (share of flags that are real fraud)")
    plt.title("Precision-Recall Curve — Fraud Detection (5-fold CV, out-of-fold)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(proba_by_model, y_true, save_path):
    plt.figure(figsize=(7, 6))
    for name, y_proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection (5-fold CV, out-of-fold)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
