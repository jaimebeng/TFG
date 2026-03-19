import os
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import linregress




class ProcessData():

    def __init__(self):
        self.output_path = "/home/jaime/Documents/TFG/data/processed"
        os.makedirs(self.output_path, exist_ok=True)
    

    # auxiliary method for hurst exponent
    def hurst(close, window=63, n_values=[6,8,11,15,21]):

        log_returns = np.log(close / close.shift(1))
        valid_idx = log_returns.notna()
        lr = log_returns[valid_idx].to_numpy()
        N = len(lr)
        
        if N < window:
            return pd.Series(np.nan, index=close.index) 

        win_arr = sliding_window_view(lr, window_shape=window)
        
        H_list = []

        for win in win_arr:
            rs_vals = []
            n_used = []
            for n in n_values:
                if n >= window:
                    continue
                chunks = window // n
                reshaped = win[:chunks*n].reshape(chunks, n)
                
                mean_chunk = reshaped.mean(axis=1, keepdims=True)
                dev = reshaped - mean_chunk
                cum_dev = np.cumsum(dev, axis=1)
                R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
                S = reshaped.std(axis=1)
                rs = R / S
                rs = rs[np.isfinite(rs)]
                if len(rs) > 0:
                    rs_vals.append(rs.mean())
                    n_used.append(n)
            
            if len(rs_vals) > 1:
                H = np.polyfit(np.log(n_used), np.log(rs_vals), 1)[0]
            else:
                H = np.nan
            H_list.append(H)
    
        H_array = np.full(len(close), np.nan)
        start_idx = (~valid_idx).sum() + (window - 1)
        H_array[start_idx:start_idx + len(H_list)] = H_list
            
        return pd.Series(H_array, index=close.index)

    def process_ticker(self, df):
        df["Returns"] = df["Close"].pct_change()
        df["Log Returns"] = np.log(df["Close"]).diff()
        df["Monthly Log Returns"] = df["Log Returns"].rolling(21).sum()
        df["Quarterly Log Returns"] = df["Log Returns"].rolling(63).sum()
        df["Semi-annual Log Returns"] = df["Log Returns"].rolling(126).sum()
        df["Parkinson Volatility"] = np.sqrt( (1 / (4*np.log(2))) * ( (np.log( df["High"]/ df["Low"])) **2 ))
        df["Monthly Volatility"] = df["Log Returns"].rolling(21).std()
        df["Quarterly Volatility"] = df["Log Returns"].rolling(63).std()
        df["Semi-annual Volatility"] = df["Log Returns"].rolling(126).std()
        df["Volatility Ratio"] = df["Monthly Volatility"] / df["Quarterly Volatility"]
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["MA20/MA60"] = df["MA20"] / df["MA60"]
        df["Distance from MA20"] = ((df["Close"] - df["MA20"]) / df["MA20"] ) * 100
        df["Distance from MA60"] = ((df["Close"] - df["MA60"]) / df["MA60"] ) * 100
        df["VMA20"] = df["Volume"].rolling(20).mean()
        df["VMA60"] = df["Volume"].rolling(60).mean()
        df["Relative Volume"] = df["Volume"] / df["VMA20"].shift(1)
        df["Volume Flow Ratio"] = df["VMA20"] / df["VMA60"]
        df["Daily Range"] = df["High"] - df["Low"]
        df["Monthly Average Intraday Range"] = df["Daily Range"].rolling(21).mean()
        df["True Range"] = np.maximum.reduce([df["Daily Range"],(df["High"] - df["Close"].shift(1)).abs(),(df["Low"] - df["Close"].shift(1)).abs()])
        df["Monthly Average True Range"] = df["True Range"].rolling(21).mean()
        df["Daily Candle Body"] = (df["Close"] - df["Open"])
        df["Monthly Average Candle Body"] = df["Daily Candle Body"].abs().rolling(21).mean()
        df["Percentage Intraday Range"] = ((df["High"] - df["Low"]) / df["Open"]) * 100
        df["Monthly Average Percentage Intraday Range"] = df["Percentage Intraday Range"].rolling(21).mean()
        df["Price Difference"] = df["Close"].diff()
        df["Gains"] = df["Price Difference"].where(df["Price Difference"] > 0, 0)
        df["Losses"] = df["Price Difference"].where(df["Price Difference"] < 0, 0).abs()
        df["Average Gains"] = df["Gains"].ewm(alpha=1/14, adjust=False).mean()
        df["Average Losses"] = df["Losses"].ewm(alpha=1/14, adjust=False).mean()
        df["RS"] = df["Average Gains"] / df["Average Losses"]
        df["RSI"] = 100 - (100 / (1 + df["RS"]))
        df["14-day Average True Range"] = df["True Range"].ewm(alpha=1/14, adjust=False).mean()
        df["Normalized Average True Range"] = (df["14-day Average True Range"]/df["Close"]) * 100
        df["10-day Efficiency Ratio"] = df["Close"].diff(10).abs() / df["Close"].diff().abs().rolling(10).sum()
        df["Monthly Efficiency Ratio"] = df["Close"].diff(21).abs() / df["Close"].diff().abs().rolling(21).sum()
        df["Body Up"] = df["Daily Candle Body"].clip(lower=0)
        df["Body Down"] = -df["Daily Candle Body"].clip(upper=0)
        epsilon = 1e-8
        rolling_up = df["Body Up"].rolling(21).mean()
        rolling_down = df["Body Down"].rolling(21).mean()
        df["Monthly Conviction Ratio"] = rolling_up / (rolling_down + epsilon)
        df["Monthly Parkinson Volatility"] = np.sqrt((1 / (4 * 21 * np.log(2))) * ((np.log(df["High"] / df["Low"])) ** 2).rolling(21).sum())
        df["Monthly Rogers-Satchell Volatility"] = np.sqrt(((np.log(df["High"]/df["Close"]) * np.log(df["High"]/df["Open"])) + (np.log(df["Low"]/df["Close"]) * np.log(df["Low"]/df["Open"]))).rolling(21).mean())
        df["Volatility of Volatility"] = df["Parkinson Volatility"].rolling(21).std()
        df["Monthly Overnight Gap Ratio"] = ((df["Open"]-df["Close"].shift(1))/df["Close"].shift(1)).rolling(21).mean()
        df["Monthly Skewness"] = df["Log Returns"].rolling(21).skew()
        df["Monthly Kurtosis"] = df["Log Returns"].rolling(21).kurt()
        df["Quarterly Hurst Exponent"] = hurst(df['Close'], window=63, n_values=[6,8,11,15,21])
        df["Quarterly Efficiency Ratio"] = df["Close"].diff(63).abs() / df["Close"].diff().abs().rolling(63).sum()
        df["Monthly Close-Location Value"] = (((2*df["Close"]) - df["High"] - df["Low"])/df["Daily Range"]).rolling(21).mean()
        df["Monthly Intraday Intensity"] = df["Monthly Close-Location Value"] * (1/df["Volume"])
        df["Monthly Normalized Intraday Intensity"] = df["Monthly Close-Location Value"] * (df["Volume"]/df["Volume"].rolling(21).mean())
        df["5-day RSI Slope"] = df['RSI'].rolling(5).apply(lambda y: linregress(range(5), y)[0], raw=True)

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

    
