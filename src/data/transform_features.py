"""
Module for cross-sectional feature transformation across multiple stock tickers.
Aggregates monthly ticker-level features and applies cross-sectional transformations
(e.g., date-by-date winsorization and standardization) to normalize inputs and target returns,
preventing temporal data leakage.
"""

import pandas as pd
from src.data.data_loader import DataLoad
from src.utils.transformer import StockTransformer



class FeatureTransformation():
    """
    Applies cross-sectional transformations to compiled ticker feature datasets.

    This class:
    1. Concatenates historical feature dataframes for all 30 tickers.
    2. Groups the unified dataframe by date to perform cross-sectional analysis.
    3. Transforms features (winsorization, z-scoring) date-by-date using `StockTransformer`
       to ensure zero mean and unit variance cross-sectionally each month.
    4. Applies similar cross-sectional normalization to target returns (`Target`).
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