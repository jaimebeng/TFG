import os
import pandas as pd
from pathlib import Path
from joblib import load
import numpy as np

class DataLoad():
    """Utility class for loading project datasets from disk.
    Supports loading a single ticker file or multiple ticker files by data type."""

    def __init__(self):
        self._data_path = "/home/jaime/Documents/TFG/data/"
    
    def load_single_data(self,type,ticker):
        if type not in ["raw","clean","processed","features"]:
            raise ValueError("Type must be raw, clean, processed or features")
        
        path = os.path.join(self._data_path, type, f"{ticker}.csv")
        df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
        print(f"{ticker}.csv loaded succesfully")

        return df
    
    def load_multiple_data(self,type):
        if type not in ["raw","clean","processed","features"]:
            raise ValueError("Type must be raw, clean, processed or features")

        directory = os.path.join(self._data_path, type)
        dfs = {}
        for filename in os.listdir(directory):
            if filename in ["GSPC.csv", "market_caps.csv", "IRX.csv"]:
                continue
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
            file = Path(filename)
            ticker = file.stem
            dfs[ticker] = df
            print(f"{ticker}.csv loaded succesfully")
            
        return dfs
    
    def load_market_caps(self, type):
        if type not in ["raw","processed"]:
            raise ValueError("Type must be raw or processed")
        file_path = os.path.join("/home/jaime/Documents/TFG/data", type, "market_caps.csv")
        df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
        print("market_caps.csv loaded succesfully")

        return df
    
    def load_risk_free_rate(self, type):
        if type not in ["raw"]:
            raise ValueError("Type must be raw ")
        file_path = os.path.join("/home/jaime/Documents/TFG/data", type, "IRX.csv")
        df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
        print("GS3M.csv loaded succesfully")

        return df

    def load_dataset(self,type, cutoff = None, date = None):
        if type not in ["model", "hpt", "returns", "GSPC", "market_caps"]:
            raise ValueError("Type must be model, hpt, returns, GSPC or market_caps")
        if type == "model":
            path = os.path.join(self._data_path, "datasets", "model.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            print("model.csv loaded succesfully")
            return df
        elif type == "hpt":
            if cutoff == None:
                raise ValueError("HPT dataset needs cutoff data")
            path = os.path.join(self._data_path, "datasets")
            X_cache = load(f"{path}/X_cache.joblib")
            y_cache = load(f"{path}/y_cache.joblib")
            months = load(f"{path}/months.joblib")
            X, y = self._get(X_cache, y_cache, months, cutoff)
            return X, y
        elif type == "returns":
            path = os.path.join(self._data_path, "datasets", "daily_stock_returns.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            print("daily_stock_returns.csv loaded succesfully")
            path = os.path.join(self._data_path, "datasets", "monthly_stock_returns.csv")
            df2 = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            print("monthly_stock_returns.csv loaded succesfully")
            return df, df2
        elif type == "GSPC":
            path = os.path.join(self._data_path, "datasets", "GSPC.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            print("GSPC.csv loaded succesfully")
            return df
        elif type == "market_caps":
            if date == None:
                raise ValueError("Market caps dataset needs a date")
            path = os.path.join(self._data_path, "datasets", "market_caps.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            df = df[df.index == date]
            return df

    def _get_index(self, months, cutoff_date):
        cutoff_date = np.datetime64(cutoff_date)

        idx = np.searchsorted(months, cutoff_date, side="right") - 1

        if idx < 0:
            raise ValueError("Cutoff date before dataset start")

        return idx

    def _get(self, X_cache, y_cache, months, cutoff):
        idx = self._get_index(months, cutoff)

        X = X_cache[:idx+1]
        y = y_cache[:len(X_cache[idx])]

        return X, y
