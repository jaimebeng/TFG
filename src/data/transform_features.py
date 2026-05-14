import os
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoad
from src.utils.transformer import StockTransformer
from sklearn.base import clone
from joblib import dump


class FeatureTransformation():
    """Loads feature data, applies cross-sectional-level transformations by date, and
    writes the transformed dataset to data/transformed/Dataset.csv.
    """

    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/data/transformed"
        os.makedirs(self._output_path, exist_ok=True)


    def transform_features(self):
        dl = DataLoad()
        dfs = dl.load_multiple_data("features")
        df = pd.concat(dfs.values()).sort_index()
        features = [c for c in df.columns if c not in ["Ticker","Target"]]
        trans = StockTransformer()
        df[features] = df.groupby(df.index, group_keys=False)[features].apply(trans.transform_x)
        df["Target"] = df.groupby(df.index, group_keys=False)["Target"].apply(trans.transform_y)
        full_path = os.path.join(self._output_path, "Dataset.csv")
        df.to_csv(full_path, index=True, date_format="%Y-%m-%d")
        print("Dataset created succesfully")
    