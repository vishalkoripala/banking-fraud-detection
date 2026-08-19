"""
explain.py

SHAP-based explainability for the deployed fraud pipeline, in plain, human
units rather than scaled/capped ones.

Why this is model-agnostic: train_final_model.py picks whichever model
(logistic regression or gradient boosting) wins on the train-split CV, and
that choice can change if the pipeline is retrained (different data,
different split). Explainability shouldn't silently break -- or silently
mislead -- just because the winning model type changed. FraudExplainer
inspects the fitted classifier and picks the matching exact SHAP method:
LinearExplainer for linear models, TreeExplainer for tree ensembles.

Both give EXACT Shapley values for their respective model family (no
sampling/approximation), and both are fast enough to run per-request.

Output units: SHAP values here are in margin (log-odds) space -- additive
contributions to the model's raw decision function, before the sigmoid.
That's the standard, robust way to attribute tree/linear model output
(shap "probability" mode exists but is slower and, for tree ensembles,
only approximate). We report direction and relative ranking rather than
claiming an exact probability decomposition, which is the honest framing
of what SHAP margin values are.
"""
import re

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def _clean_feature_name(name: str) -> str:
    """ColumnTransformer prefixes names like 'amount__Amount' -> 'Amount'."""
    return re.sub(r"^[a-z]+__", "", name)


class FraudExplainer:
    """Wraps a fitted (preprocess -> classify) Pipeline with SHAP explanations."""

    def __init__(self, fitted_pipeline, background_X: pd.DataFrame):
        self.pipeline = fitted_pipeline
        self.preprocessor = fitted_pipeline.named_steps["preprocess"]
        self.classifier = fitted_pipeline.named_steps["classify"]
        self.feature_names = [_clean_feature_name(n) for n in self.preprocessor.get_feature_names_out()]

        background_transformed = self.preprocessor.transform(background_X)

        if isinstance(self.classifier, LogisticRegression):
            self.model_family = "linear"
            # 100 background rows is plenty to estimate a stable expected-value
            # baseline for a linear model; set explicitly so it's a deliberate
            # choice rather than a silently-applied library default.
            masker = shap.maskers.Independent(background_transformed, max_samples=100)
            self.explainer = shap.LinearExplainer(self.classifier, masker)
        else:
            self.model_family = "tree"
            self.explainer = shap.TreeExplainer(self.classifier)

    def explain_instance(self, raw_features: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """
        raw_features: single-row DataFrame in the pipeline's original column
        order/units (i.e. exactly what the API receives, pre-preprocessing).
        Returns the top_n features ranked by |SHAP contribution|, each with
        the RAW (human-readable) value alongside the contribution -- so the
        response says "Amount = $4,812.50 -> increases risk", not
        "amount__Amount = 3.42 -> increases risk".
        """
        transformed = self.preprocessor.transform(raw_features)
        shap_values = self.explainer.shap_values(transformed)
        if isinstance(shap_values, list):  # some SHAP paths return a list per class
            shap_values = shap_values[-1]
        contributions = np.asarray(shap_values)[0]

        raw_row = raw_features.iloc[0]
        rows = []
        for fname, contribution in zip(self.feature_names, contributions):
            rows.append({
                "feature": fname,
                "value": round(float(raw_row[fname]), 4),
                "contribution": round(float(contribution), 4),
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
            })
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows[:top_n]

    def global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP value| per feature across a sample -- overall model behavior."""
        transformed = self.preprocessor.transform(X_sample)
        shap_values = self.explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
