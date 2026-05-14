from src.data.download_data import DownloadData
from src.data.clean_data import CleanData
from src.data.process_data import ProcessData
from src.data.create_features import FeatureCreation
from src.data.transform_features import FeatureTransformation
from src.data.hpt_dataset import HPTDataset
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


DOWNLOAD = False
CLEAN = False
PROCESS = False
FEATURE = False
TRANSFORM = False
DATASET = True
tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AVGO","AMZN", "HD", "MCD", 
           "NKE","JNJ", "UNH", "PFE", "ABBV","CAT", "BA", "UPS", "MMM","XOM", 
           "CVX", "SLB","JPM", "BAC", "GS", "MS","NEE", "LIN", "DOW","VZ", "CMCSA"
]

def main():
    if DOWNLOAD:
        dd = DownloadData()
        dd.download_tick_data(tickers)
        dd.download_gspc()
        print("Data downloaded successfully.")
    if CLEAN:
        cd = CleanData()
        cd.clean_data()
        print("Data cleaned successfully.")
    if PROCESS:
        pd = ProcessData()
        pd.process_data()
        print("Data processed successfully.")
    if FEATURE:
        fc = FeatureCreation()
        fc.create_features()
        print("Features engineered successfully.")
    if TRANSFORM:
        ft = FeatureTransformation()
        ft.transform_features()
        print("Features transformed successfully.")
    if DATASET:
        hpt = HPTDataset()
        hpt.create_ds()




if __name__ == "__main__":
    main()