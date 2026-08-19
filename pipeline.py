"""
pipeline.py

Builds the full fraud-detection preprocessing + modeling pipeline for the
ULB Credit Card Fraud Detection dataset (columns: Time, V1-V28, Amount, Class).

Design decisions worth calling out
-----------------------------------
1. Outlier capping is applied ONLY to `Amount`, not to V1-V28.
   V1-V28 are already PCA components, and fraudulent transactions are
   disproportionately found in the tails of those components -- that's
   part of what makes them predictive. Capping them would clip away the
   exact signal the model needs. `Amount`, on the other hand, is raw and
   right-skewed (a handful of very large legitimate purchases can dominate
   a distance/scale-sensitive model), so it's the one column worth taming.

2. RobustScaler (median/IQR based) is used instead of StandardScaler
   everywhere, since it stays stable in the presence of the outliers this
   project is explicitly about handling -- a few extreme values can't drag
   the mean/std around the way they would with StandardScaler.

3. Every statistic (impute medians, cap bounds, scaler center/scale) is
   learned inside a Pipeline's .fit(), so when cross_val_predict refits
   this pipeline per fold, preprocessing is fit on that fold's training
   rows only. No test-fold information leaks into preprocessing.

4. Class imbalance (~0.17% fraud) is handled via class_weight="balanced"
   rather than resampling (e.g. SMOTE). This keeps the pipeline simple and
   leakage-proof by construction -- there's no synthetic-sample step that
   could accidentally run before the train/test split.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from transformers import OutlierCapper

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
TIME_COLUMN = ["Time"]
AMOUNT_COLUMN = ["Amount"]


def build_preprocessor():
    """Returns a fresh (unfit) ColumnTransformer for the three column groups."""

    amount_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap", OutlierCapper(iqr_multiplier=3.0)),  # tame extreme purchase amounts only
        ("scale", RobustScaler()),
    ])

    time_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])

    pca_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),  # no capping here -- see module docstring, point 1
    ])

    return ColumnTransformer([
        ("amount", amount_pipeline, AMOUNT_COLUMN),
        ("time", time_pipeline, TIME_COLUMN),
        ("pca", pca_pipeline, PCA_COLUMNS),
    ])


def build_model_pipelines(random_state=42):
    """
    Returns {model_name: full sklearn Pipeline}, each with its OWN fresh
    preprocessor instance (important -- sharing one ColumnTransformer object
    between pipelines would mean they silently share fitted state).
    """
    return {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("classify", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocess", build_preprocessor()),
            ("classify", HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }
