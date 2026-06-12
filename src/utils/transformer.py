"""
Module for feature scaling and outlier handling in cross-sectional datasets.
Provides transformation utilities like winsorization and z-score standardization
to process feature and target matrices cross-sectionally on each individual date.
"""

from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd


class StockTransformer():
    """
    Applies winsorization and z-score standardization to a cross-section of assets.

    Used date-by-date to scale features and target variables:
    1. Winsorizes feature values within specified percentile limits (default: 1% lower and upper tails)
       to limit the impact of extreme statistical outliers.
    2. Performs standard scaling (z-score normalization) to obtain zero mean and unit variance.
    3. Applied cross-sectionally (i.e. grouped by date) to normalize values across all active stocks
       on each individual timestamp, avoiding temporal leakage.
    """

    def __init__(self, winsor_limits=(0.01, 0.01)):
        self.winsor_limits = winsor_limits

    def transform_x(self, X):

        X = X.copy()

        for col in X.columns:
            X[col] = np.asarray(winsorize(X[col].values, limits=self.winsor_limits), dtype=float)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=X.columns)

        return X_scaled

    def transform_y(self,y):

        y = y.copy()
        scaler = StandardScaler()
        y_scaled = pd.Series(scaler.fit_transform(y.values.reshape(-1, 1)).ravel(),index=y.index,name=y.name)

        return y_scaled

