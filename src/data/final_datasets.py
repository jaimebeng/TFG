import numpy as np
import os
from src.data.data_loader import DataLoad
from src.data.transform_features import FeatureTransformation
from joblib import dump
import pandas as pd
import pandas_market_calendars as mcal


class Datasets():
    """
    """

    def __init__(self, order):
        self._path = "/home/jaime/Documents/TFG/data/datasets"
        os.makedirs(self._path, exist_ok=True)
        self._dl = DataLoad()
        self._order = sorted(order)
    
    def create_model_dataset(self):
        dfs = self._dl.load_multiple_data("features")
        ft = FeatureTransformation()
        df = ft.transform_features(dfs)
        full_path = os.path.join(self._path, "model.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Backtest dataset created succesfully")

    def create_hpt_dataset(self):
        df = self._dl.load_dataset("model")
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

    def create_returns_dataset(self):
        dfs = self._dl.load_multiple_data("processed")
        for tick in dfs.keys():
            dfs[tick] = dfs[tick][["Returns"]].copy()
            dfs[tick].rename(columns={"Returns": tick},inplace=True)
            dfs[tick].dropna(how='any', inplace=True)
        df = pd.concat(dfs.values(),axis=1).sort_index()
        df = df[(df.index >= "2010-12-31") & (df.index <= "2025-12-31")]
        df = df[self._order]
        full_path = os.path.join(self._path, "daily_stock_returns.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        dfs = self._dl.load_multiple_data("processed")
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date="1960-01-01",end_date="2030-12-31")
        month_ends = (schedule.index.to_series().groupby(schedule.index.to_period("M")).max())
        for tick in dfs.keys():
            months = dfs[tick].index.to_period("M")
            dfs[tick].index = pd.DatetimeIndex(months.map(month_ends), name=dfs[tick].index.name)
            dfs[tick]["Returns"] = dfs[tick]["Close"].pct_change()
            dfs[tick] = dfs[tick][["Returns"]].copy()
            dfs[tick].rename(columns={"Returns": tick},inplace=True)
            dfs[tick].dropna(how='any', inplace=True)
        df = pd.concat(dfs.values(),axis=1).sort_index()
        df = df[(df.index >= "2010-12-31") & (df.index <= "2025-12-31")]
        df = df[self._order]
        df = df[self._order]
        full_path = os.path.join(self._path, "monthly_stock_returns.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Returns datasets created succesfully")

    def create_snp500_dataset(self):
        df = self._dl.load_single_data("processed", "GSPC")
        df["Returns"] = df["Close"].pct_change()
        df = df[["Returns"]]
        df = df[(df.index >= "2010-12-31") & (df.index <= "2025-12-31")]
        full_path = os.path.join(self._path, "GSPC.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("S&P500 dataset created succesfully")

    def create_market_caps_dataset(self):
        df = self._dl.load_market_caps("processed")
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date="1960-01-01",end_date="2030-12-31")
        month_ends = (schedule.index.to_series().groupby(schedule.index.to_period("M")).max())
        months = df.index.to_period("M")
        df.index = pd.DatetimeIndex(months.map(month_ends), name=df.index.name)
        df = df[(df.index >= "2010-12-31") & (df.index <= "2025-12-31")]
        df = df[self._order]
        full_path = os.path.join(self._path, "market_caps.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Market caps dataset created succesfully")

    def create_fama_dataset(self):
        df = self._dl.load_fama("clean")
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date="1960-01-01",end_date="2030-12-31")
        month_ends = (schedule.index.to_series().groupby(schedule.index.to_period("M")).max())
        months = df.index.to_period("M")
        df.index = pd.DatetimeIndex(months.map(month_ends), name=df.index.name)
        df = df[(df.index >= "2010-12-31") & (df.index <= "2025-12-31")]
        full_path = os.path.join(self._path, "fama.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Fama french 5 factors dataset created succesfully")