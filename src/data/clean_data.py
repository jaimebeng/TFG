import pandas as pd
import os

class CleanData():
    """Minimal data cleaner for already well-structured input files.
    Applies only lightweight column/index formatting adjustments.
    Saves in data/clean.
    """

    def __innit__(self,dfs):
        self.dfs = dfs
        self.output_path = "/Users/jaime/Documents/UPM/TFG/data/clean"
        os.makedirs(self.output_path, exist_ok=True)
    
    def clean_data(self):

        for ticker,df in self.dfs.items():
            df = df.droplevel(1,axis=1)
            df.columns.name = None
            file_path = os.path.join(self.output_path, f"{ticker}.csv")
            df.to_csv(file_path,index=True) 

