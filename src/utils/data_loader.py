import os
import pandas as pd
from pathlib import Path

class DataLoader():
    """Utility class for loading project datasets from disk.
    Supports loading a single ticker file or multiple ticker files by data type."""

    def __init__(self):
        self.data_path = "/home/jaime/Documents/TFG/data/"
    
    def load_single_data(self,type,ticker):
        if type not in ["raw","clean","processed","features"]:
            raise ValueError("Type must be raw, clean, processed or features")
        
        path = os.path.join(self.data_path, type, f"{ticker}.csv")
        df = pd.read_csv(path, header=0, index_col=0)
        print(f"{ticker}.csv loaded succesfully")

        return df
    
    def load_multiple_data(self,type):
        if type not in ["raw","clean","processed","features"]:
            raise ValueError("Type must be raw, clean, processed or features")

        directory = os.path.join(self.data_path, type)
        dfs = {}
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=0, index_col=0)
            file = Path(filename)
            ticker = file.stem
            dfs[ticker] = df
            print(f"{ticker}.csv loaded succesfully")
            
        return dfs

