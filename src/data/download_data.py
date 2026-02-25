import yfinance as yf
import os

class DownloadData():
    """Downloads historical stock price data from yfinance and saves it as CSV files."""

    def __init__(self,):
        self.output_path = "/Users/jaime/Documents/UPM/TFG/data/raw"
        os.makedirs(self.output_path, exist_ok=True)

    def download_tick_data(self,tickers):

        for tick in tickers: 
            file_path = os.path.join(self.output_path, f"{tick}.csv")
            print(f" Downloading {tick}") 
            try: 
                df = yf.download(tick, start="2010-01-01", end="2026-02-01",auto_adjust=False) 
                df.to_csv(file_path,index=True) 
            except Exception as e:
                print(f"Error downloading {tick}: {e}") 
        
        print("\Download complete.")

    def download_gspc(self):
        file_path = f"{self.output_path}/GSPC.csv"
        print(" Downloading GSPC") 
        try: 
            df = yf.download("^GSPC", start="2010-01-01", end="2026-02-01",auto_adjust=False) 
            df.to_csv(file_path,index=True) 
        except Exception as e:
            print("Error downloading GSPC: {e}") 
        
        print("\nDownload complete.")