"""
tests/test_pipeline.py

Unit tests for transformers.py, pipeline.py, and evaluate.py. These use
small synthetic data and don't require the trained model artifacts --
run with: pytest tests/test_pipeline.py
"""
import numpy as np
import pandas as pd
import pytest

from transformers import OutlierCapper
from pipeline import build_preprocessor, build_model_pipelines, PCA_COLUMNS
from evaluate import best_f1_threshold, summarize_model


# --------------------------------------------------------------------------
# OutlierCapper
# --------------------------------------------------------------------------
def test_outlier_capper_clips_extreme_values():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(1000, 1))
    X[0, 0] = 500.0  # inject an obvious outlier

    capper = OutlierCapper(iqr_multiplier=3.0).fit(X)
    transformed = capper.transform(X)

    assert transformed[0, 0] < 500.0
    assert transformed[0, 0] == pytest.approx(capper.upper_[0])


def test_outlier_capper_leaves_typical_values_unchanged():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(1000, 1))

    capper = OutlierCapper(iqr_multiplier=3.0).fit(X)
    transformed = capper.transform(X)

    # The vast majority of a roughly-normal sample should be untouched by a 3xIQR cap
    unchanged_fraction = np.isclose(transformed, X).mean()
    assert unchanged_fraction > 0.98


def test_outlier_capper_bounds_come_only_from_fit_data():
    """Bounds must be frozen at fit time -- transform() must not look at new data's range."""
    train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    capper = OutlierCapper(iqr_multiplier=1.5).fit(train)

    bounds_before = (capper.lower_.copy(), capper.upper_.copy())
    unseen = np.array([[9999.0], [-9999.0]])
    capper.transform(unseen)
    bounds_after = (capper.lower_.copy(), capper.upper_.copy())

    assert np.array_equal(bounds_before[0], bounds_after[0])
    assert np.array_equal(bounds_before[1], bounds_after[1])


def test_outlier_capper_feature_names_pass_through():
    capper = OutlierCapper().fit(np.zeros((5, 3)))
    names = capper.get_feature_names_out(["a", "b", "c"])
    assert list(names) == ["a", "b", "c"]


# --------------------------------------------------------------------------
# Preprocessor / pipeline construction
# --------------------------------------------------------------------------
def _synthetic_transactions(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {"Time": rng.uniform(0, 172792, n), "Amount": rng.exponential(50, n)}
    for col in PCA_COLUMNS:
        data[col] = rng.normal(0, 1, n)
    X = pd.DataFrame(data)
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.95, 0.05]))
    return X, y


def test_preprocessor_output_shape_matches_input_columns():
    X, _ = _synthetic_transactions()
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape == (len(X), X.shape[1])  # capping/scaling never changes column count


def test_preprocessor_handles_missing_values():
    X, _ = _synthetic_transactions()
    X.loc[0, "Amount"] = np.nan
    X.loc[1, "V1"] = np.nan
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert not np.isnan(transformed).any()


def test_build_model_pipelines_returns_expected_models():
    pipelines = build_model_pipelines()
    assert set(pipelines.keys()) == {"logistic_regression", "gradient_boosting"}
    for pipe in pipelines.values():
        assert hasattr(pipe, "fit") and hasattr(pipe, "predict_proba")


def test_fitted_pipeline_predicts_valid_probabilities():
    X, y = _synthetic_transactions(n=300)
    pipe = build_model_pipelines()["logistic_regression"]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# Evaluation utilities
# --------------------------------------------------------------------------
def test_best_f1_threshold_is_a_valid_probability():
    rng = np.random.default_rng(2)
    y_true = rng.choice([0, 1], size=500, p=[0.9, 0.1])
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, 500), 0, 1)

    threshold, f1 = best_f1_threshold(y_true, y_proba)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_summarize_model_confusion_matrix_totals_match_n_samples():
    rng = np.random.default_rng(3)
    y_true = rng.choice([0, 1], size=200, p=[0.9, 0.1])
    y_proba = rng.uniform(0, 1, 200)

    summary = summarize_model("test_model", y_true, y_proba, threshold=0.5)
    assert summary["confusion_matrix"].sum() == 200
    assert 0.0 <= summary["average_precision"] <= 1.0
    assert 0.0 <= summary["roc_auc"] <= 1.0
