from sklearn.base import BaseEstimator, TransformerMixin
from numba import njit
import numpy as np


@njit
def winsorize_and_standardize(data, lower, upper):
    n, m = data.shape
    out = np.empty_like(data)

    for j in range(m):
        col = data[:, j].copy()
        sorted_col = np.sort(col)

        low_val = sorted_col[int(n * lower)]
        high_val = sorted_col[int(n * (1 - upper)) - 1]

        for i in range(n):
            if col[i] < low_val:
                col[i] = low_val
            elif col[i] > high_val:
                col[i] = high_val

        mean = col.mean()
        std = col.std()
        if std == 0:
            std = 1e-8

        for i in range(n):
            out[i, j] = (col[i] - mean) / std

    return out

class PerStockTransformer(BaseEstimator, TransformerMixin):
    """Applies per-stock winsorization and standardization for sklearn pipelines.
    Independently processes each stock's features to handle outliers and normalize values.
    """
    def __init__(self, stock_col='Ticker', winsor_limits=(0.01, 0.01)):
        self.stock_col = stock_col
        self.winsor_limits = winsor_limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reset_index(drop=True)

        features = [c for c in X.columns if c != self.stock_col]
        X_arr = X[features].to_numpy(dtype=np.float64)
        transformed = np.empty_like(X_arr, dtype=np.float64)

        lower, upper = self.winsor_limits

        groups = X.groupby(self.stock_col).groups

        for _, idx in groups.items():
            idx = np.array(list(idx))
            data = X_arr[idx, :]

            transformed[idx, :] = winsorize_and_standardize(
                data, lower, upper
            )

        return transformed
