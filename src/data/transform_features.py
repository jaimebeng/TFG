import pandas as pd
from src.data.data_loader import DataLoad
from src.utils.transformer import StockTransformer



class FeatureTransformation():
    """
    """

    def __init__(self):
        pass

    def transform_features(self, dfs):
        df = pd.concat(dfs.values()).sort_index()
        features = [c for c in df.columns if c not in ["Ticker","Target"]]
        trans = StockTransformer()
        df[features] = df.groupby(df.index, group_keys=False)[features].apply(trans.transform_x)
        df["Target"] = df.groupby(df.index, group_keys=False)["Target"].apply(trans.transform_y)
        print("Features transformed succesfully")
        return df