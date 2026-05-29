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
            if filename in ["market_caps.csv", "fama.csv"]:
                continue
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
    
    def clean_fama_data(self):
        directory = "/home/jaime/Documents/TFG/data/raw"
        full_path = os.path.join(directory, "fama.csv")
        df = pd.read_csv(full_path)
        df.drop(columns="Unnamed: 0.1", inplace=True)
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df = df[df["Date"].astype(str).str.match(r"^\d{6}$")]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
        df.set_index("Date", inplace=True)
        df = df.astype(float)
        df = df / 100
        file_path = os.path.join(self._output_path, "fama.csv")
        df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
        print("fama.csv cleaned succesfully")



