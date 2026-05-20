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
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
            file = Path(filename)
            ticker = file.stem
            dfs[ticker] = df
            print(f"{ticker}.csv loaded succesfully")
            
        return dfs
    
    def load_backtest(self,type, cutoff = None):
        if type not in ["backtest", "hpt", "optimisation"]:
            raise ValueError("Type must be backtest, hpt or optimisation")
        if type == "backtest":
            path = os.path.join(self._data_path, "datasets", "backtest.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            print("backtest.csv loaded succesfully")
            return df
        elif type == "hpt":
            path = os.path.join(self._data_path, "datasets")
            X_cache = load(f"{path}/X_cache.joblib")
            y_cache = load(f"{path}/y_cache.joblib")
            months = load(f"{path}/months.joblib")
            X, y = self._get(X_cache, y_cache, months, cutoff)
            return X, y
        else:
            path = os.path.join(self._data_path, "optimisation", "backtest.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            print("optimisation.csv loaded succesfully")
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




