import os
import pandas as pd
from pathlib import Path
from joblib import load

class DataLoad():
    """Utility class for loading project datasets from disk.
    Supports loading a single ticker file or multiple ticker files by data type."""

    def __init__(self):
        self._data_path = "/home/jaime/Documents/TFG/data/"
    
    def load_single_data(self,type,ticker):
        if type not in ["raw","clean","processed","features","transformed"]:
            raise ValueError("Type must be raw, clean, processed, features or transformed")
        
        path = os.path.join(self._data_path, type, f"{ticker}.csv")
        df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
        print(f"{ticker}.csv loaded succesfully")

        return df
    
    def load_multiple_data(self,type):
        if type not in ["raw","clean","processed","features","transformed"]:
            raise ValueError("Type must be raw, clean, processed, features or transformed")

        directory = os.path.join(self._data_path, type)
        dfs = {}
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=0, index_col=0, parse_dates=True)
            file = Path(filename)
            ticker = file.stem
            dfs[ticker] = df
            print(f"{ticker}.csv loaded succesfully")
            
        return dfs




