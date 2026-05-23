import yfinance as yf
import os
import re
import json
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd

class DownloadData():
    """Downloads historical stock price data from yfinance and saves it as CSV files."""

    def __init__(self,):
        self._output_path = "/home/jaime/Documents/TFG/data/raw"
        os.makedirs(self._output_path, exist_ok=True)

    def download_tick_data(self,tickers, start_date="2010-01-01", end_date="2026-02-01"):

        for tick in tickers: 
            file_path = os.path.join(self._output_path, f"{tick}.csv")
            print(f"\nDownloading {tick}") 
            try: 
                df = yf.download(tick, start=start_date, end=end_date,auto_adjust=True) 
                df.to_csv(file_path,index=True) 
            except Exception as e:
                print(f"\nError downloading {tick}: {e}") 
        
        print("\nDownload complete.")

    def download_gspc(self, start_date="2010-01-01", end_date="2026-02-01"):
        file_path = f"{self._output_path}/GSPC.csv"
        print("\nDownloading GSPC") 
        try: 
            df = yf.download("^GSPC", start=start_date, end=end_date,auto_adjust=True) 
            df.to_csv(file_path,index=True) 
        except Exception as e:
            print("\nError downloading GSPC: {e}") 
        
        print("\nDownload complete.")

    def _extract_market_cap_data(self, ticker, slug, start_date="2010-01-01", end_date="2026-02-01"):
        url = f"https://companiesmarketcap.com/{slug}/marketcap/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        print(f"\nDownloading market cap for {ticker}")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"\nError: Received status code {response.status_code} for {ticker}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            data_found = False
            
            for script in soup.find_all('script'):
                script_text = script.string or ''
                if 'data = [' in script_text:
                    match = re.search(r'data\s*=\s*(\[.*?\]);', script_text, re.DOTALL)
                    if match:
                        raw_data = json.loads(match.group(1))
                        parsed_data = []
                        for entry in raw_data:
                            if 'd' in entry and 'm' in entry:
                                dt = datetime.fromtimestamp(entry['d'])
                                date_str = dt.strftime('%Y-%m-%d')
                                if start_date <= date_str <= end_date:
                                    market_cap_usd = int(entry['m']) * 100000
                                    parsed_data.append({
                                        'date': date_str,
                                        'market_cap': market_cap_usd
                                    })
                        data_found = True
                        return parsed_data
                        
            if not data_found:
                print(f"\nError: Could not find chart data in the HTML page for {ticker}")
                
        except Exception as e:
            print(f"\nException occurred while processing {ticker}: {str(e)}")
            
        return None

    def download_market_caps(self):  

        TICKER_SLUGS = {'AAPL': 'apple','MSFT': 'microsoft','NVDA': 'nvidia','GOOG': 'alphabet-google','ORCL': 'oracle',
            'AVGO': 'broadcom','AMZN': 'amazon','HD': 'home-depot','MCD': 'mcdonald','NKE': 'nike','JNJ': 'johnson-and-johnson',
            'UNH': 'united-health','PFE': 'pfizer','MRK': 'merck','CAT': 'caterpillar','BA': 'boeing','UPS': 'ups','MMM': '3m',
            'XOM': 'exxon-mobil','CVX': 'chevron','SLB': 'schlumberger','JPM': 'jp-morgan-chase','BAC': 'bank-of-america',
            'GS': 'goldman-sachs','MS': 'morgan-stanley','NEE': 'nextera-energy','LIN': 'linde','SHW': 'sherwin-williams',
            'VZ': 'verizon','CMCSA': 'comcast'
        }        
        
        market_cap_frames = []
        
        for ticker, slug in TICKER_SLUGS.items():
            
            ticker_data = self._extract_market_cap_data(ticker, slug)
            
            if ticker_data:
                ticker_df = pd.DataFrame(ticker_data)
                ticker_df["date"] = pd.to_datetime(ticker_df["date"])
                ticker_df = ticker_df.set_index("date")["market_cap"].rename(ticker)
                market_cap_frames.append(ticker_df)
                print(f"\nDownload complete for {ticker}.")
            else:
                print(f"\nFailed to extract data for {ticker}")
                
            time.sleep(1.5)

        if market_cap_frames:
            market_caps_df = pd.concat(market_cap_frames, axis=1).sort_index()
            market_caps_df.index.name = "Date"
            output_file = os.path.join(self._output_path, "market_caps.csv")
            market_caps_df.to_csv(output_file, index=True)
            print(f"\nSaved market caps to {output_file}")
