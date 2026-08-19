"""
transformers.py

Custom scikit-learn-compatible transformers for the fraud detection pipeline.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Winsorizes (caps) extreme values column-wise using the IQR method.

    Why a custom transformer instead of just clipping the raw data by hand:
    wiring this into a Pipeline/ColumnTransformer means the cap bounds are
    learned with .fit() on the TRAINING fold only. When the pipeline is
    refit inside cross-validation, each fold computes its own bounds from
    its own training data -- the held-out fold's values never influence
    where the caps are drawn. That is what "safely" means in this project:
    outlier handling that can't leak test-set information into training.

    Parameters
    ----------
    iqr_multiplier : float, default=3.0
        Multiplier applied to the IQR to set the lower/upper fences.
        1.5 = classic Tukey fence (aggressive, flags mild outliers too).
        3.0 = conservative fence, only reins in genuinely extreme values.
        Financial "Amount" data has plenty of legitimate large purchases,
        so this pipeline defaults to the more conservative 3.0.
    """

    def __init__(self, iqr_multiplier=3.0):
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_multiplier * iqr
        self.upper_ = q3 + self.iqr_multiplier * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # Capping doesn't add/remove/rename columns, so pass names through.
        # Needed for sklearn's set_output(transform="pandas") / pipeline introspection.
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features)
