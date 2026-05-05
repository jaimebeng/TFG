import os
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoad
from src.utils.transformer import StockTransformer
from sklearn.base import clone
from joblib import dump


class FeatureTransformation():
    """Builds and stores rolling transformed feature matrices.
    Applies per-stock transformation incrementally to avoid recomputing for each model.
    Saves in data/transformed.
    """

    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/data/transformed"
        os.makedirs(self._output_path, exist_ok=True)


    def transform_features(self):
        dl = DataLoad()
        dfs = dl.load_multiple_data("features")
        df = pd.concat(dfs.values()).sort_index()
        X = df.drop(columns="Target")
        y = df["Target"].reset_index(drop=True)
        months = np.sort(X.index.unique())
        X_cache = []
        transformer = StockTransformer()
        for month in months:
            trans = clone(transformer)
            X_slice = X[X.index <= month].reset_index(drop=True)
            Xt = trans.fit_transform(X_slice)
            X_cache.append(Xt)
        dump(X_cache, os.path.join(self._output_path, "X_cache.joblib"))
        dump(y.to_numpy(copy=True), os.path.join(self._output_path, "y_aligned.joblib"))        
        dump(months, os.path.join(self._output_path, "months.joblib"))

        

    