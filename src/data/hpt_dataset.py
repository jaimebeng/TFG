import numpy as np
import os
from src.data.data_loader import DataLoad
from joblib import dump, load


class HPTDataset():
    """Loads cached transformed data and returns the correct cutoff slice.
    Provides quick access to the features/targets pair up to a given date.
    Used for hyperparameter tuning.
    """

    def __init__(self):
        self._path = "/home/jaime/Documents/TFG/data/hpt"
        os.makedirs(self._path, exist_ok=True)
        if len(os.listdir(self._path)) == 0:
            self.create_ds()
        self._X_cache = load(f"{self._path}/X_cache.joblib")
        self._y = load(f"{self._path}/y_cache.joblib")
        self._months = load(f"{self._path}/months.joblib")
    
    def create_ds(self):
        dl = DataLoad()
        df = dl.load_single_data("transformed","Dataset")
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
        
    def _get_index(self, cutoff_date):
        cutoff_date = np.datetime64(cutoff_date)

        idx = np.searchsorted(self._months, cutoff_date, side="right") - 1

        if idx < 0:
            raise ValueError("Cutoff date before dataset start")

        return idx

    def get(self, cutoff):
        idx = self._get_index(cutoff)

        X = self._X_cache[:idx+1]
        y = self._y[:len(self._X_cache[idx])]

        return X, y