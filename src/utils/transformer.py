from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd


class StockTransformer():
    """Applies winsorization and standardization to the whole group of stocks.
    Normalizes each feature across all data to handle outliers and standardize values globally.
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

