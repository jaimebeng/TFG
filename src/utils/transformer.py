from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import numpy as np


class StockTransformer(BaseEstimator, TransformerMixin):
    """Applies winsorization and standardization to the whole group of stocks.
    Normalizes each feature across all data to handle outliers and standardize values globally.
    """
    def __init__(self, winsor_limits=(0.01, 0.01)):
        self.winsor_limits = winsor_limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        for col in X.columns:
            X[col] = np.asarray(winsorize(X[col].values, limits=self.winsor_limits), dtype=float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled