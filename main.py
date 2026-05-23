from src.data.download_data import DownloadData
from src.data.clean_data import CleanData
from src.data.process_data import ProcessData
from src.data.create_features import FeatureCreation
from src.data.transform_features import FeatureTransformation
from src.data.final_datasets import Datasets
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


DOWNLOAD = False
CLEAN = False
PROCESS = False
FEATURE = True
DATASET = True
tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "ORCL", "AVGO","AMZN", "HD", "MCD", 
           "NKE","JNJ", "UNH", "PFE", "MRK","CAT", "BA", "UPS", "MMM","XOM", 
           "CVX", "SLB","JPM", "BAC", "GS", "MS","NEE", "LIN", "SHW","VZ", "CMCSA"]

def main():
    if DOWNLOAD:
        dd = DownloadData()
        dd.download_tick_data(tickers)
        dd.download_gspc()
        dd.download_market_caps()
        print("Data downloaded successfully.")
    if CLEAN:
        cd = CleanData()
        cd.clean_data()
        print("Data cleaned successfully.")
    if PROCESS:
        pd = ProcessData()
        pd.process_data()
        pd.process_market_caps()
        print("Data processed successfully.")
    if FEATURE:
        fc = FeatureCreation()
        fc.create_features()
        print("Features engineered successfully.")
    if DATASET:
        ds = Datasets()
        ds.create_backtest_dataset()
        ds.create_hpt_dataset()
        ds.create_returns_dataset()
        ds.create_market_caps_dataset()
        print("Datasets created successfully.")




if __name__ == "__main__":
    main()