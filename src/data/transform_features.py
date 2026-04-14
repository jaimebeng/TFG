import os
import pandas as pd
from data.data_loader import DataLoad
from utils.transformer import PerStockTransformer
from sklearn.base import clone
from joblib import dump


class FeatureTransformation():

    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/data/transformed"
        os.makedirs(self._output_path, exist_ok=True)

    def _dataset_joiner(self,dfs):
        for ticker, df_stock in dfs.items():
            df_stock['Ticker'] = ticker

        df_all = pd.concat(dfs.values()).sort_index()

        return df_all


    def transform_features(self):
        dl = DataLoad()
        dfs = dl.load_multiple_data("features")
        df = self._dataset_joiner(dfs)
        X = df.drop(columns="Target")
        X_cache = []
        transformer = PerStockTransformer()
        for t in range(1, len(X)):
            trans = clone(transformer)
            Xt = trans.fit_transform(X.iloc[:t+1].reset_index(drop=True))
            X_cache.append(Xt)

        cache_path = os.path.join(self._output_path, "X_cache.joblib")
        dump(X_cache, cache_path)
        

