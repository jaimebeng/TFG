import yfinance as yf
import os

class DownloadData():

    def __init__(self,tickers):
        self.tickers = tickers
        self.output_path = "/Users/jaime/Documents/UPM/TFG/data/raw"
        os.makedirs(self.output_path, exist_ok=True)

    def download_tick_data(self):

        for tick in self.tickers: 
            file_path = os.path.join(self.output_path, f"{tick}.csv")
            print(f" Descargando {tick}") 
            try: 
                df = yf.download(tick, start="2010-01-01", end="2026-02-01",auto_adjust=False) 
                df.to_csv(file_path,index=True) 
            except Exception as e:
                print(f"Error descargando {tick}: {e}") 
        
        print("\nDescarga completada.")

    def download_spy(self):
        file_path = f"{self.output_path}/spy.csv"
        print(" Descargando SPY") 
        try: 
            df = yf.download("SPY", start="2010-01-01", end="2026-02-01",auto_adjust=False) 
            df.to_csv(file_path,index=True) 
        except Exception as e:
            print("Error descargando SPY: {e}") 
        
        print("\nDescarga completada.")

    def download_gspc(self):
        file_path = f"{self.output_path}/gspc.csv"
        print(" Descargando GSPC") 
        try: 
            df = yf.download("^GSPC", start="2010-01-01", end="2026-02-01",auto_adjust=False) 
            df.to_csv(file_path,index=True) 
        except Exception as e:
            print("Error descargando GSPC: {e}") 
        
        print("\nDescarga completada.")