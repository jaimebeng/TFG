import os
import pandas as pd
import numpy as np
from pathlib import Path
import pandas_market_calendars as mcal


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
        nyse = mcal.get_calendar('NYSE')
        nyse_bme = pd.offsets.CustomBusinessMonthEnd(calendar=nyse.regular_holidays)
        df = df.resample(nyse_bme).last()
        df["Target"] = (np.log(df["Close"] / df["Close"].shift(1))).shift(-1)
        df.dropna(subset=["Target"], inplace=True)
        features = [
            'High', 'Low', 'Open', 'Returns',
            'MA20', 'Close', 'Monthly Average Percentage Intraday Range', 'True Range',
            'Daily Range', 'Monthly Average Intraday Range', '14-day Average True Range', 'Percentage Intraday Range',
            'Quarterly Volatility', 'Monthly Average Candle Body', 'Average Gains', 'Average Losses',
            'MA60', 'VMA20', 'Volume', 'RS',
            'Monthly Log Returns', 'Zscore Price', 'High Low Trend Position', 'Volatility Breakout',
            'Monthly Parkinson Volatility', 'Monthly Volatility', 'Monthly Normalized Intraday Intensity', 'MA20 Alpha',
            'Quarterly Alpha', 'Quarterly Log Returns', 'Monthly Return Momentum', 'Semi-annual Alpha',
            '126 Day Beta', 'Semi-annual Volatility', 'Quarterly Alpha Volatility',
            "Distance from MA60", "Daily Candle Body", "Normalized Average True Range", "Upside Volatility", 
            "RSI", "Downside Volatility", "63 Day Beta", "Log Returns",
            "Price Difference", "Distance from MA20", "Volume Flow Ratio", "Gains",
            "Losses", "Body Up", "Monthly Efficiency Ratio", "Volatility Term Structure Slope",
            "Volatility Regime Change", "Monthly Skewness", "Quarterly Efficiency Ratio", "Return Autocorr 5D",
            "Return Autocorr 10D", "Gap Direction", "Monthly Close-Location Value", "Gap Persistence",
            "Volume Weighted Return", "Signed Volume Pressure", "Dollar Volume", "21 Day Beta",
            "Mean Reversion Pressure", "Up Days Ratio", "Daily Alpha",
            "Amihud_Illiquidity", "K_Ratio_21d", "VPIN_Proxy"
        ]
        
        df = df.drop(columns=features)

        df = df.copy()
        
        return df


    def create_features(self):
        directory = "/home/jaime/Documents/TFG/data/processed"
        for filename in os.listdir(directory):
            if filename not in ["GSPC.csv","market_caps.csv","GS3M.csv"]:
                full_path = os.path.join(directory, filename)
                df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
                file = Path(filename)
                ticker = file.stem
                df = self._features_engineering(df,ticker)
                file_path = os.path.join(self._output_path, f"{ticker}.csv")
                df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
                print(f"{ticker}.csv feature engineered succesfully")