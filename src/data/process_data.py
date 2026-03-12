import os
import numpy as np
import pandas as pd
from pathlib import Path


class ProcessData():

    def __init__(self):
        self.output_path = "/home/jaime/Documents/TFG/data/processed"
        os.makedirs(self.output_path, exist_ok=True)

    def process_ticker(self, df):
        df["Returns"] = df["Adj Close"].pct_change()
        df["Log Returns"] = np.log(df["Adj Close"]).diff()
        df["Monthly Log Returns"] = df["Log Returns"].rolling(21).sum()
        df["Quarterly Log Returns"] = df["Log Returns"].rolling(63).sum()
        df["Semi-annual Log Returns"] = df["Log Returns"].rolling(126).sum()
        df["Daily Volatility"] = np.sqrt( (1 / (4*np.log(2))) * ( (np.log( df["High"]/ df["Low"])) **2 ))
        df["Monthly Volatility"] = df["Log Returns"].rolling(21).std()
        df["Quarterly Volatility"] = df["Log Returns"].rolling(63).std()
        df["Semi-annual Volatility"] = df["Log Returns"].rolling(126).std()
        df["Volatility Ratio"] = df["Monthly Volatility"] / df["Quarterly Volatility"]
        df["MA20"] = df["Log Returns"].rolling(20).mean()
        df["MA60"] = df["Log Returns"].rolling(60).mean()
        df["MA20/MA60"] = df["MA20"] / df["MA60"]
        df["Distance from MA20"] = ((df["Adj Close"] - df["MA20"]) / df["MA20"] ) * 100
        df["Distance from MA60"] = ((df["Adj Close"] - df["MA20"]) / df["MA60"] ) * 100
        df["VMA20"] = df["Volume"].rolling(20).mean()
        df["VMA60"] = df["Volume"].rolling(60).mean()
        df["Relative Volume"] = df["Volume"] / df["VMA20"].shift(1)
        df["Volume Flow Ratio"] = df["VMA20"] / df["VMA60"]
        df["Daily Range"] = df["High"] - df["Low"]
        df["Monthly Average Intraday Range"] = df["Daily Range"].rolling(21).mean()
        df["True Range"] = np.maximum.reduce([df["Daily Range"],(df["High"] - df["Adj Close"].shift(1)).abs(),(df["Low"] - df["Adj Close"].shift(1)).abs()])
        df["Monthly Average True Range"] = df["True Range"].rolling(21).mean()
        df["Daily Candle Body"] = (df["Adj Close"] - df["Open"])
        df["Monthly Average Candle Body"] = df["Daily Candle Body"].abs().rolling(21).mean()
        df["Percentage Intraday Range"] = ((df["High"] - df["Low"]) / df["Open"]) * 100
        df["Monthly Average Percentage Intraday Range"] = df["Percentage Intraday Range"].rolling(21).mean()
        df["Price Difference"] = df["Adj Close"].diff()
        df["Gains"] = df["Price Difference"].where(df["Price Difference"] > 0, 0)
        df["Losses"] = df["Price Difference"].where(df["Price Difference"] < 0, 0).abs()
        df["Average Gains"] = df["Gains"].rolling(14).mean()
        df["Average Losses"] = df["Losses"].rolling(14).mean()
        df["Average Gains"] = df["Gains"].ewm(alpha=1/14, adjust=False).mean()
        df["Average Losses"] = df["Losses"].ewm(alpha=1/14, adjust=False).mean()
        df["RS"] = df["Average Gains"] / df["Average Losses"]
        df["RSI"] = 100 - (100 / (1 + df["RS"]))
        df["14-day Average True Range"] = df["True Range"].ewm(alpha=1/14, adjust=False).mean()
        df["Normalized Average True Range"] = (df["14-day Average True Range"]/df["Adj Close"]) * 100
        df["10-day Efficiency Ratio"] = df["Adj Close"].diff(10).abs() / df["Adj Close"].diff().abs().rolling(10).sum()
        df["10-day Efficiency Ratio"] = df["Adj Close"].diff(21).abs() / df["Adj Close"].diff().abs().rolling(21).sum()
        df["Body Up"] = df["Daily Candle Body"].clip(lower=0)
        df["Body Down"] = -df["Daily Candle Body"].clip(upper=0)
        epsilon = 1e-8
        rolling_up = df["Body Up"].rolling(21).mean()
        rolling_down = df["Body Down"].rolling(21).mean()
        df["Monthly Conviction Ratio"] = rolling_up / (rolling_down + epsilon)

        return df

    def process_data(self):
        directory = "/home/jaime/Documents/TFG/data/clean"
        for filename in os.listdir(directory):
            if filename != "GSPC.csv":
                full_path = os.path.join(directory, filename)
                df = pd.read_csv(full_path, header=0, index_col=0)
                df = self.process_ticker(df)
                file = Path(filename)
                ticker = file.stem
                file_path = os.path.join(self.output_path, f"{ticker}.csv")
                df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
                print(f"{ticker}.csv processed succesfully")

    
