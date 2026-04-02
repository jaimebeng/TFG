import pandas as pd
import os
from pathlib import Path


class CleanData():
    """Minimal data cleaner for already well-structured input files.
    Applies only lightweight column/index formatting adjustments.
    Saves in data/clean.
    """

    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/data/clean"
        os.makedirs(self._output_path, exist_ok=True)
    
    def clean_data(self):
        directory = "/home/jaime/Documents/TFG/data/raw"
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path, header=[0,1], index_col=0)
            df = df.droplevel(1,axis=1)
            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            file = Path(filename)
            ticker = file.stem
            file_path = os.path.join(self._output_path, f"{ticker}.csv")
            df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
            print(f"{ticker}.csv cleaned succesfully")


