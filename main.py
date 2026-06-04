from src.data.download_data import DownloadData
from src.data.clean_data import CleanData
from src.data.process_data import ProcessData
from src.data.create_features import FeatureCreation
from src.data.final_datasets import Datasets
from src.backtest.backtest import Backtest
import numpy as np
import torch

rng = np.random.default_rng(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOWNLOAD = False
CLEAN = False
PROCESS = False
FEATURE = False
DATASET = False
BACKTEST = True

tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "ORCL", "AVGO","AMZN", "HD", "MCD", 
           "NKE","JNJ", "UNH", "PFE", "MRK","CAT", "BA", "UPS", "MMM","XOM", 
           "CVX", "SLB","JPM", "BAC", "GS", "MS","NEE", "LIN", "SHW","VZ", "CMCSA"]

def main():
    if DOWNLOAD:
        dd = DownloadData()
        dd.download_tick_data(tickers)
        dd.download_gspc()
        dd.download_market_caps()
        dd.download_fama_data()
        print("Data downloaded successfully.")
    if CLEAN:
        cd = CleanData()
        cd.clean_data()
        cd.clean_fama_data()
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
        ds = Datasets(tickers)
        ds.create_model_dataset()
        ds.create_hpt_dataset()
        ds.create_returns_dataset()
        ds.create_snp500_dataset()
        ds.create_market_caps_dataset()
        ds.create_fama_dataset()
        print("Datasets created successfully.")
    if BACKTEST:
        backtest = Backtest()
        backtest.snp500_backtest()
        print("S&P500 backtest complete")
        backtest.eq_portfolio_backtest()
        print("Equally Weighted Portfolio backtest complete")
        backtest.mvo_portfolio_backtest()
        print("Mean Variance Optimization Portfolio backtest complete")
        backtest.portfolio_backtest("No Model Predictions Black Litterman")
        print("No Model Black Litterman backtest complete")
        backtest.portfolio_backtest("Random Predictions", "random")
        print("Random Predictions backtest complete")
        backtest.portfolio_backtest("Linear Regression", "linreg")
        print("Linear Regression backtest complete")
        backtest.portfolio_backtest("Lasso Regression", "lassoreg", grid=1)
        print("Lasso Regression backtest complete")
        backtest.portfolio_backtest("Ridge Regression", "ridgereg", grid=1)
        print("Ridge Regression backtest complete")
        backtest.portfolio_backtest("ElasticNet Regression", "elasticnet", grid=1)
        print("ElasticNet Regression backtest complete")
        backtest.portfolio_backtest("Random Forest", "rf", grid=1)
        print("Random Forest backtest complete")
        backtest.portfolio_backtest("XGBoost", "xgboost", grid=1)
        print("XGBoost backtest complete")
        backtest.pytorch_portfolio_backtest("MLP", "mlp", device=device)
        print("MLP backtest complete")
        backtest.pytorch_portfolio_backtest("CNN", "cnn", device=device)
        print("CNN backtest complete")
        backtest.pytorch_portfolio_backtest("LSTM", "lstm", device=device)
        print("LSTM backtest complete")
        backtest.pytorch_portfolio_backtest("CNN + LSTM", "cnn_lstm", device=device)
        print("CNN + LSTM backtest complete")
        backtest.pytorch_portfolio_backtest("Transformer", "transformer", device=device)
        print("CNN backtest complete")
        print("Backtests execution finished successfully")

if __name__ == "__main__":
    main()