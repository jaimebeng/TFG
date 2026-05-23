import os
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import linregress
import pandas_market_calendars as mcal



class ProcessData():
    """Computes technical indicators and price-based features for stock data.
    Generates returns, volatility, moving averages, RSI, and Hurst exponent metrics.
    Saves in data/processed.
    """

    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/data/processed"
        os.makedirs(self._output_path, exist_ok=True)
    

    def _hurst(self, close, window=63, n_values=None):
        if n_values is None:
            n_values = [6, 8, 11, 15, 21]
        
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
                if chunks < 2:
                    continue  
                reshaped = win[:chunks * n].reshape(chunks, n)

                mean_chunk = reshaped.mean(axis=1, keepdims=True)
                dev = reshaped - mean_chunk
                cum_dev = np.cumsum(dev, axis=1)
                R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
                S = reshaped.std(axis=1, ddof=1) 
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
    
    def _calculate_k_ratio(self,series):
        x = np.arange(len(series))
        y = series.cumsum()
        slope, _, _, _, std_err = linregress(x, y)
        return slope / std_err if std_err != 0 else 0

    def _process_ticker(self, df, gspc):
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
        df["Downside Volatility"] = df["Log Returns"].clip(upper=0).rolling(21).std()
        df["Upside Volatility"] = df["Log Returns"].clip(lower=0).rolling(21).std()
        df["Volatility Term Structure Slope"] = df["Monthly Volatility"] / df["Semi-annual Volatility"]
        df["Volatility Breakout"] = df["Monthly Volatility"] / df["Monthly Volatility"].rolling(63).mean()
        df["Volatility Regime Change"] = df["Monthly Volatility"].diff(21)
        df["Monthly Skewness"] = df["Log Returns"].rolling(21).skew()
        df["Monthly Kurtosis"] = df["Log Returns"].rolling(21).kurt()
        df["Return Autocorr 5D"] = df["Log Returns"].rolling(21).apply(lambda x: x.autocorr(lag=1))
        df["Return Autocorr 10D"] = df["Log Returns"].rolling(21).apply(lambda x: x.autocorr(lag=2))
        df["Entropy of Returns"] = df["Log Returns"].rolling(21).apply(lambda x: -np.sum(np.histogram(x, bins=10, density=True)[0] * np.log(np.histogram(x, bins=10, density=True)[0] + 1e-8)))
        df["Monthly Return Momentum"] = df["Log Returns"].rolling(21).sum()
        df["Monthly Return Acceleration"] = df["Monthly Return Momentum"].diff()
        df["Trend Strength"] = abs(df["MA20/MA60"] - 1)
        df["High Low Trend Position"] = (df["Close"] - df["Low"].rolling(21).min()) /(df["High"].rolling(21).max() - df["Low"].rolling(21).min())
        df["Quarterly Efficiency Ratio"] = df["Close"].diff(63).abs() / df["Close"].diff().abs().rolling(63).sum()
        df["Monthly Hurst Exponent"] = self._hurst(df['Close'], window=21, n_values=[6,8,11,15,21])
        df["Quarterly Hurst Exponent"] = self._hurst(df['Close'], window=63, n_values=[6,8,11,15,21])
        df["5-day RSI Slope"] = df['RSI'].rolling(5).apply(lambda y: linregress(range(5), y)[0], raw=True)
        df["ADX Proxy"] = (abs(df["High"].diff()) + abs(df["Low"].diff())).rolling(21).mean() / df["True Range"].rolling(21).mean()
        df["Gap Up"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        df["Gap Direction"] = np.sign(df["Gap Up"])
        df["Gap Persistence"] = df["Gap Up"].rolling(5).mean()
        df["Gap vs Intraday Move"] = df["Gap Up"] / (df["High"] - df["Low"])
        df["Monthly Close-Location Value"] = (((2*df["Close"]) - df["High"] - df["Low"])/df["Daily Range"]).rolling(21).mean()
        df["Zscore Price"] = (df["Close"] - df["Close"].rolling(21).mean()) / df["Close"].rolling(21).std()
        df["Mean Reversion Pressure"] = -df["Zscore Price"] * df["RSI"]
        df["Up Days Ratio"] = (df["Log Returns"] > 0).rolling(21).mean()
        df["Dollar Volume"] = df["Close"] * df["Volume"]
        df["Volume Weighted Return"] = df["Log Returns"] * df["Volume"]
        df["Signed Volume Pressure"] = np.sign(df["Log Returns"]) * df["Volume"]
        df["Cumulative Volume Trend"] = df["Volume"].rolling(21).sum() / df["Volume"].rolling(63).sum()
        df["Price Volume Correlation"] = df["Log Returns"].rolling(21).corr(df["Volume"])
        df["Monthly Intraday Intensity"] = df["Monthly Close-Location Value"] * (1/df["Volume"])
        df["Monthly Normalized Intraday Intensity"] = df["Monthly Close-Location Value"] * (df["Volume"]/df["Volume"].rolling(21).mean())
        df["Zscore Volume"] = (df["Volume"] - df["Volume"].rolling(21).mean()) / df["Volume"].rolling(21).std()
        df["21 Day Beta"] = df["Log Returns"].rolling(21).cov(gspc["Log Returns"]) / gspc["21 Day Var"]
        df["63 Day Beta"] = df["Log Returns"].rolling(63).cov(gspc["Log Returns"]) / gspc["63 Day Var"]
        df["126 Day Beta"] = df["Log Returns"].rolling(126).cov(gspc["Log Returns"]) / gspc["126 Day Var"]
        df["MA 20 Beta"] = df["21 Day Beta"].rolling(21).mean()
        df["MA 63 Beta"] = df["63 Day Beta"].rolling(63).mean()
        df["MA 126 Beta"] = df["126 Day Beta"].rolling(126).mean()
        df["Daily Alpha"] = df["Log Returns"] - (df["21 Day Beta"] * gspc["Log Returns"])
        df["Monthly Alpha"] = df["Daily Alpha"].rolling(21).sum()
        df["Quarterly Alpha"] = df["Daily Alpha"].rolling(63).sum()
        df["Semi-annual Alpha"] = df["Daily Alpha"].rolling(126).sum()
        df["MA20 Alpha"] = df["Daily Alpha"].rolling(20).mean()
        df["MA60 Alpha"] = df["Daily Alpha"].rolling(60).mean()
        df["Monthly Alpha Volatility"] = df["Daily Alpha"].rolling(21).std()
        df["Quarterly Alpha Volatility"] = df["Daily Alpha"].rolling(63).std()
        df["Semi-annual Alpha Volatility"] = df["Daily Alpha"].rolling(126).std()
        df["Amihud_Illiquidity"] = df["Log Returns"].abs() / (df["Dollar Volume"] + 1e-8)
        df["Amihud_Illiquidity_21d"] = df["Amihud_Illiquidity"].rolling(21).mean()
        df["K_Ratio_21d"] = df["Log Returns"].rolling(21).apply(self._calculate_k_ratio, raw=True)
        range_daily = (df["High"] - df["Low"]) + 1e-8
        df["Volume_Imbalance"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / range_daily
        df["VPIN_Proxy"] = df["Volume_Imbalance"] * df["Volume"]
        df["VPIN_Signal_21d"] = df["VPIN_Proxy"].rolling(21).mean()
        
        df = df.copy()

        return df

    def _process_gspc(self,gspc):
        gspc["Returns"] = gspc["Close"].pct_change()
        gspc["Log Returns"] = np.log(gspc["Close"]).diff()
        gspc["21 Day Var"] = gspc["Log Returns"].rolling(21).var()
        gspc["63 Day Var"] = gspc["Log Returns"].rolling(63).var()
        gspc["126 Day Var"] = gspc["Log Returns"].rolling(126).var()

        return gspc

    def process_data(self):
        directory = "/home/jaime/Documents/TFG/data/clean"
        gspc = pd.read_csv(os.path.join(directory, "GSPC.csv"), header=0, index_col=0, parse_dates=True)
        gspc = self._process_gspc(gspc)
        file_path = os.path.join(self._output_path, "GSPC.csv")
        gspc.to_csv(file_path, index=True, date_format="%Y-%m-%d")
        print("GSPC.csv processed succesfully")
        for filename in os.listdir(directory):
            if filename != "GSPC.csv":
                full_path = os.path.join(directory, filename)
                df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
                df = self._process_ticker(df,gspc)
                file = Path(filename)
                ticker = file.stem
                file_path = os.path.join(self._output_path, f"{ticker}.csv")
                df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
                print(f"{ticker}.csv processed succesfully")

    def process_market_caps(self):
        path = os.path.join("/home/jaime/Documents/TFG/data/raw", "market_caps.csv")
        df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
        nyse = mcal.get_calendar("NYSE")

        all_days = pd.date_range(start="2010-01-01", end="2025-12-31", freq="D")
        all_days = pd.DatetimeIndex(all_days)
        if getattr(all_days, 'tz', None) is not None:
            all_days = all_days.tz_convert("UTC").tz_localize(None)

        nyse_days = nyse.valid_days(start_date="2010-01-01", end_date="2025-12-31")
        nyse_days = pd.DatetimeIndex(nyse_days)
        if getattr(nyse_days, 'tz', None) is not None:
            nyse_days = nyse_days.tz_convert("UTC").tz_localize(None)

        df.index = pd.to_datetime(df.index)
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()
        df = df.reindex(all_days)

        for col in df.columns:
            col_first = df[col].first_valid_index()
            if col_first is not None:
                df.loc[:col_first, col] = df.loc[col_first, col]

        df = df.ffill()
        df = df.reindex(nyse_days)

        file_path = os.path.join(self._output_path, "market_caps.csv")
        df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
        print("market_caps.csv processed succesfully")



    
