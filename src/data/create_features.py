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
        features = [
            'High', 'Low', 'Open', 'Returns', 'MA60',
            'Close', 'Normalized Average True Range', 'Percentage Intraday Range', 'Daily Range', 'True Range',
            'Monthly Average True Range', 'Monthly Average Intraday Range', 'Quarterly Volatility', 'Monthly Average Candle Body', 'Average Gains',
            'Average Losses', '14-day Average True Range', 'VMA20', 'Volume', 'RS',
            "Zscore Price", "Mean Reversion Pressure", "Monthly Parkinson Volatility", "Monthly Rogers-Satchell Volatility", "Entropy of Returns",
            "Cumulative Volume Trend", "Volatility Breakout", "Monthly Normalized Intraday Intensity", "Zscore Volume", "MA20 Alpha",
            "Quarterly Alpha", "Semi-annual Alpha", "126 Day Beta", "Semi-annual Volatility", "Quarterly Alpha Volatility",
            "Distance from MA60", "Daily Candle Body", "Monthly Volatility", "Upside Volatility", "Monthly Average Percentage Intraday Range",
            "RSI", "Downside Volatility", "Monthly Log Returns", "Quarterly Log Returns", "Volatility Ratio",
            "Log Returns", "MA20/MA60", "Losses", "Volatility Term Structure Slope", "Return Autocorr 10D",
            "Trend Strength", "Monthly Return Momentum", "Monthly Kurtosis", "Gap Up", "Up Days Ratio",
            "Gap vs Intraday Move", "Volume Weighted Return", "Price Volume Correlation", "Price Difference", "Quarterly Hurst Exponent",
            "Volatility Regime Change", "Quarterly Efficiency Ratio", "21 Day Beta", "Signed Volume Pressure", "Monthly Close-Location Value",
            "Daily Alpha", "K_Ratio_21d", "Volume_Imbalance", "VPIN_Signal_21d"
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