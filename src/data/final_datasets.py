import numpy as np
import os
from src.data.data_loader import DataLoad
from src.data.transform_features import FeatureTransformation
from joblib import dump, load
import pandas as pd


class Datasets():
    """
    """

    def __init__(self):
        self._path = "/home/jaime/Documents/TFG/data/datasets"
        os.makedirs(self._path, exist_ok=True)
        self.dl = DataLoad()
    
    def create_backtest_dataset(self):
        ft = FeatureTransformation()
        df = ft.transform_features()
        full_path = os.path.join(self._path, "backtest.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Backtest dataset created succesfully")

    def create_hpt_dataset(self):
        df = self.dl.load_backtest("backtest")
        X = df.drop(columns=["Ticker","Target"])
        y = df["Target"].to_numpy()
        months = np.sort(X.index.unique())
        X_cache = []
        for month in months:
            X_slice = X[X.index <= month].to_numpy()
            X_cache.append(X_slice)
        dump(X_cache, os.path.join(self._path, "X_cache.joblib"))
        dump(y, os.path.join(self._path, "y_cache.joblib"))
        dump(months, os.path.join(self._path, "months.joblib"))
        print("Hyperparameter tuning dataset created")

    def create_optimisation_dataset(self):
        dfs = self.dl.load_multiple_data("processed")
        for tick in dfs.keys():
            dfs[tick] = dfs[tick][["Log Returns"]].copy()
            dfs[tick].rename(columns={"Log Returns": tick},inplace=True)
            dfs[tick].dropna(how='any', inplace=True)
        df = pd.concat(dfs.values(),axis=1).sort_index()
        full_path = os.path.join(self._path, "optimisation.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Optimisation dataset created succesfully")