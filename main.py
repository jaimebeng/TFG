from src.data.download_data import DownloadData
from src.data.clean_data import CleanData
from src.data.process_data import ProcessData


DOWNLOAD = False
CLEAN = False
PROCESS = False
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



if __name__ == "__main__":
    main()