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

    def _features_engineering(self,df,ticker):
        df["Ticker"] = ticker
        df.dropna(how='any', inplace=True)
        df = df.resample("BME").last()
        df["Target"] = (np.log(df["Close"] / df["Close"].shift(1))).shift(-1)
        df.dropna(subset=["Target"], inplace=True)
        features = ["High", "Low", "Open", "Close", "Returns",
                    "MA20", "Normalized Average True Range", "Daily Range", "14-day Average True Range", "Monthly Average True Range",
                    "Monthly Average Intraday Range", "Parkinson Volatility", "Quarterly Volatility", "Monthly Average Candle Body", "VMA20",
                    "Volume", "RSI", "Zscore Price", "Mean Reversion Pressure", "Monthly Parkinson Volatility",
                    "Monthly Rogers-Satchell Volatility", "Volatility Breakout", "Monthly Normalized Intraday Intensity", "Relative Volume", "Quarterly Alpha",
                    "Quarterly Log Returns", "Semi-annual Alpha", "MA20 Alpha", "MA 63 Beta", "Semi-annual Volatility",
                    "Quarterly Alpha Volatility",
                    "Distance from MA60", "Daily Candle Body", "MA60", "True Range", "Monthly Average Percentage Intraday Range",
                    "Upside Volatility", "Downside Volatility", "126 Day Beta",
                    "Distance from MA20", "Volatility Ratio", "Monthly Log Returns", "Monthly Volatility", "Log Returns",
                    "Volume Flow Ratio", "Losses", "Monthly Efficiency Ratio", "Volatility Regime Change", "Quarterly Efficiency Ratio",
                    "Monthly Return Momentum", "Monthly Skewness", "Monthly Kurtosis", "Gap Direction", "Quarterly Hurst Exponent",
                    "5-day RSI Slope", "Signed Volume Pressure", "MA20/MA60", "Price Difference", "Gains",
                    "Entropy of Returns", "ADX Proxy", "Monthly Close-Location Value", "Gap vs Intraday Move", "High Low Trend Position",
                    "Up Days Ratio", "Zscore Volume", "Cumulative Volume Trend", "Daily Alpha", "Monthly Alpha","K_Ratio_21d", "VPIN_Proxy"
                    ]
        df = df.drop(columns=features)

        df = df.copy()
        
        return df


    def create_features(self):
        directory = "/home/jaime/Documents/TFG/data/processed"
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
            file = Path(filename)
            ticker = file.stem
            df = self._features_engineering(df,ticker)
            file_path = os.path.join(self._output_path, f"{ticker}.csv")
            df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
            print(f"{ticker}.csv feature engineered succesfully")