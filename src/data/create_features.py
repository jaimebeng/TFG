import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class FeatureCreation():
    """Feature engineering and selection for stock data.
    Applies resampling, drops underperforming features, and applies log transformations.
    Saves in data/features.
    """

    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/data/features"
        os.makedirs(self._output_path, exist_ok=True)

    def _features_engineering(self,df):
        df.dropna(how='any', inplace=True)
        df = df.resample("BME").last()
        df["Target"] = (np.log(df["Close"] / df["Close"].shift(1))).shift(-1)
        df.dropna(subset=["Target"], inplace=True)
        columns_to_drop = ['Close', 'High', 'Low', 'Open',
                           'MA20', 'MA60', 'True Range', 'Daily Range',
                           'Daily Candle Body', 'Price Difference', 'Gains', 'Losses',
                           'Average Gains', 'Average Losses', 'RS', 'Returns',
                           'Monthly Average Intraday Range', 'Monthly Average True Range', 
                           'Monthly Average Candle Body', 'Monthly Average Percentage Intraday Range',
                           'VMA60', 'Parkinson Volatility', 'Quarterly Volatility', 'Distance from MA20','Volume', 
                           'Distance from MA60', 'Monthly Log Returns', 'Volume Flow Ratio','10-day Efficiency Ratio',
                           '14-day Average True Range', 'Monthly Volatility', 'Monthly Rogers-Satchell Volatility',
                           'Monthly Normalized Intraday Intensity', 'Monthly Parkinson Volatility', 'Body Down', 
                           'Quarterly Hurst Exponent', '5-day RSI Slope', 'Monthly Hurst Exponent']

        df = df.drop(columns=columns_to_drop)
        skewed = df.skew() >= 1
        non_negative = (df >= 0).all()
        features = df.columns[skewed & non_negative].tolist()
        df[features] = np.log1p(df[features])

        return df


    def create_features(self):
        directory = "/home/jaime/Documents/TFG/data/processed"
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
            df = self._features_engineering(df)
            file = Path(filename)
            ticker = file.stem
            file_path = os.path.join(self._output_path, f"{ticker}.csv")
            df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
            print(f"{ticker}.csv feature engineered succesfully")