import numpy as np
from joblib import load

class Dataset:
    """Loads cached transformed data and returns the correct cutoff slice.
    Provides quick access to the feature/target pair for a given date.
    """

    def __init__(self):
        self._path = "/home/jaime/Documents/TFG/data/transformed"
        self.X_cache = load(f"{self._pathpath}/X_cache.joblib")
        self.y = load(f"{self._path}/y_aligned.joblib")
        self.months = load(f"{self._path}/months.joblib")

    def _get_index(self, cutoff_date):
        cutoff_date = np.datetime64(cutoff_date)

        idx = np.searchsorted(self.months, cutoff_date, side="right") - 1

        if idx < 0:
            raise ValueError("Cutoff date before dataset start")

        return idx

    def get(self, cutoff):
        idx = self._get_index(cutoff)

        X = self.X_cache[idx]
        y = self.y[:len(X)]

        return X, y