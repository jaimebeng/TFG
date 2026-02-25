import os
import pandas as pd

class DataLoader():
    """Utility class for loading project datasets from disk.
    Supports loading a single ticker file or multiple ticker files by data type."""

    def __init__(self):
        self.data_path = "/Users/jaime/Documents/UPM/TFG/data/"
    
    def load_single_data(self,type,ticker):
        if type not in ["raw","clean","processed","features"]:
            raise ValueError("Type must be raw, clean, processed or features")
        
        path = os.path.join(self.data_path, type, f"{ticker}.csv")
        df = pd.read_csv(path, header=[0,1], index_col=0)

        return df
    
    def load_multiple_data(self,type,tickers):
        if type not in ["raw","clean","processed","features"]:
            raise ValueError("Type must be raw, clean, processed or features")
        
        dfs = {}
        for ticker in tickers:
            path = os.path.join(self.data_path, type, f"{ticker}.csv")
            df = pd.read_csv(path, header=[0,1], index_col=0)
            dfs[ticker] = df

        return dfs

