from src.data.data_loader import DataLoad
from src.utils.plotting import Plotter
from src.models.create_models import lin_reg
from src.models.create_hpt_search import *
from src.portfolio.bl_optimization import BlackLitterman
from src.backtest.monte_carlo import MonteCarlo
from src.utils.metrics import cagr
import pandas as pd
import numpy as np

class Backtest():

    def __init__(self):
        self._dl = DataLoad()
        self._backtest = self._dl.load_dataset("model")
        self._backtest = self._backtest[self._backtest.index <= "2025-12-31"]
        self._daily_stock_returns, self._monthly_stock_returns = self._dl.load_dataset("returns")
        self._daily_stock_returns = self._daily_stock_returns[self._daily_stock_returns.index <= "2025-12-31"]
        self._monthly_stock_returns = self._monthly_stock_returns[self._monthly_stock_returns.index <= "2025-12-31"]
        self._backtest_months = self._backtest[self._backtest.index >= "2017-01-01"].index.unique()
        self._first_month = self._backtest_months[0]
        self._backtest_days = self._daily_stock_returns[self._daily_stock_returns.index > self._first_month].index.unique()
        self._first_day = self._backtest[self._backtest.index <= self._backtest_days[0]].index[-1]
        self._snp500 = self._dl.load_dataset("GSPC")
        f = self._dl.load_dataset("fama")
        self._fama = f[(f.index > self._first_month) & (f.index <= "2025-12-31")]
        self._rfr = self._fama["RF"]
        self._plotter = Plotter()
        self._monte_carlo = MonteCarlo()

    def snp500_backtest(self):
        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        total_portfolio_value = 1.0
        monthly_start_value = 1.0

        for day in self._backtest_days:

            r = float(self._snp500.loc[day].item())
            daily_returns.append(r)
            total_portfolio_value = total_portfolio_value * (1 + r)
            daily_portfolio_values.append(total_portfolio_value)

            if day in self._backtest_months:
                monthly_return = (total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(total_portfolio_value)
                monthly_start_value = total_portfolio_value

        self._snp500_daily_returns = daily_returns
        self._snp500_monthly_returns = monthly_returns
        self._snp500_cagr = cagr(total_portfolio_value, len(self._backtest_months) / 12)

        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)

        self._plotter.plot_equity_cruve("S&P 500", daily_res, self._snp500_daily_returns)
        self._plotter.plot_metrics("S&P 500", 0, monthly_returns, self._rfr, daily_portfolio_values, total_portfolio_value, monthly_portfolio_values, self._backtest_months)
        self._plotter.plot_dd("S&P 500", daily_portfolio_values, daily_res)
        
    def eq_portfolio_backtest(self):
        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        total_portfolio_value = 1
        monthly_start_value = 1
        weights = np.ones(30) / 30
        positions = weights * total_portfolio_value

        for day in self._backtest_days:
            
            r = self._daily_stock_returns.loc[day].values 
            pnl = np.sum(positions * r)
            portfolio_return = pnl / total_portfolio_value
            daily_returns.append(portfolio_return)
            total_portfolio_value += pnl
            daily_portfolio_values.append(total_portfolio_value)
            positions = positions * (1 + r)
            weights = positions / total_portfolio_value

            if day in self._backtest_months:
                monthly_return = (total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(total_portfolio_value)
                monthly_start_value = total_portfolio_value
                weights = np.ones(30) / 30
                positions = weights * total_portfolio_value

        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)
        
        self._plotter.plot_equity_cruve("Equally Weighted Portfolio", daily_res, self._snp500_daily_returns)
        self._plotter.plot_metrics("Equally Weighted Portfolio", 1, monthly_returns, self._rfr, daily_portfolio_values, total_portfolio_value, monthly_portfolio_values, self._backtest_months, self._fama)
        self._plotter.plot_dd("Equally Weighted Portfolio", daily_portfolio_values, daily_res)

        self._monte_carlo.monte_carlo_path_sim("Equally Weighted Portfolio", daily_res, self._rfr, self._snp500_cagr, self._backtest_months)

    def mvo_portfolio_backtest(self):
        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        total_portfolio_value = 1
        monthly_start_value = 1
        bl = BlackLitterman()
        first_day = self._backtest[self._backtest.index <= self._backtest_days[0]].index[-1]
        preds = pd.DataFrame({"Ticker": self._backtest.loc[first_day]["Ticker"]})
        preds["Predictions"] = self._monthly_stock_returns.loc[first_day].values
        weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= first_day].tail(504), preds=preds)
        positions = weights * total_portfolio_value
        pred_returns = []
        true_returns = []

        for day in self._backtest_days:

            r = self._daily_stock_returns.loc[day].values 
            pnl = np.sum(positions * r)
            portfolio_return = pnl / total_portfolio_value
            daily_returns.append(portfolio_return)
            total_portfolio_value += pnl
            daily_portfolio_values.append(total_portfolio_value)
            positions = positions * (1 + r)
            weights = positions / total_portfolio_value

            if day in self._backtest_months:
                monthly_return = (total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(total_portfolio_value)
                monthly_start_value = total_portfolio_value
                pred_returns.append(expected_returns)
                true_returns.append(self._monthly_stock_returns.loc[day].values)
                preds["Predictions"] = self._monthly_stock_returns.loc[day].values
                weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= day].tail(504), preds=preds)
                positions = weights * total_portfolio_value

        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)

        self._plotter.plot_equity_cruve("MVO Portfolio", daily_res, self._snp500_daily_returns)
        self._plotter.plot_metrics("MVO Portfolio", 1, monthly_returns, self._rfr, daily_portfolio_values, total_portfolio_value, monthly_portfolio_values, self._backtest_months, self._fama)
        self._plotter.plot_dd("MVO Portfolio", daily_portfolio_values, daily_res)

        self._monte_carlo.monte_carlo_path_sim("MVO Portfolio", daily_res, self._rfr, self._snp500_cagr, self._backtest_months)

    def portfolio_backtest(self, title, model_type = 0, grid = 0):
        if model_type and model_type != "random":
            if grid:
                grid = create_search(model_type)
            else:
                model = lin_reg()

        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        total_portfolio_value = 1
        monthly_start_value = 1
        preds = None
        pred_z_scores = None
        true_z_scores = None
        if model_type:
            preds = pd.DataFrame({"Ticker": self._backtest.loc[self._first_day]["Ticker"]})
            if model_type != "random":
                features = [c for c in self._backtest.columns if c not in ["Ticker", "Target"]]
                if grid:
                    prev_day = self._first_day - pd.Timedelta(days=1)
                    X, y = self._dl.load_dataset("hpt", prev_day)
                    grid.fit(X,y)
                    model = grid.best_model_
                else:
                    model.fit(self._backtest[self._backtest.index < self._first_day][features].to_numpy(), self._backtest[self._backtest.index < self._first_day]["Target"].to_numpy())
                pred = model.predict(self._backtest.loc[self._first_day][features].values)
                pred_z_scores = [pred]
                true_z_scores = [self._backtest.loc[self._first_day]["Target"].to_numpy()]
            else:
                pred = np.random.normal(loc=0, scale=1, size=30)

            preds["Predictions"] = pred
 
        bl = BlackLitterman()
        weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= self._first_day].tail(504), self._first_day, preds=preds)
        positions = weights * total_portfolio_value
        pred_returns = []
        true_returns = []

        for day in self._backtest_days:

            r = self._daily_stock_returns.loc[day].values 
            pnl = np.sum(positions * r)
            portfolio_return = pnl / total_portfolio_value
            daily_returns.append(portfolio_return)
            total_portfolio_value += pnl
            daily_portfolio_values.append(total_portfolio_value)
            positions = positions * (1 + r)
            weights = positions / total_portfolio_value

            if day in self._backtest_months:
                monthly_return = (total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(total_portfolio_value)
                monthly_start_value = total_portfolio_value
                pred_returns.append(expected_returns)
                true_returns.append(self._monthly_stock_returns.loc[day].values)
                if model_type:
                    if model_type != "random":
                        if grid and day.month_name() == "December":
                            prev_day = day - pd.Timedelta(days=1)
                            X, y = self._dl.load_dataset("hpt", prev_day)
                            grid.fit(X,y)
                            model = grid.best_model_
                        else:
                            model.fit(self._backtest[self._backtest.index < day][features].to_numpy(), self._backtest[self._backtest.index < day]["Target"].to_numpy())

                        pred = model.predict(self._backtest.loc[day][features].values)
                        pred_z_scores.append(pred)
                        true_z_scores.append(self._backtest.loc[day]["Target"].to_numpy())
                    else:
                        pred = np.random.normal(loc=0, scale=1, size=30) 
                    preds["Predictions"] = pred
                weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= day].tail(504), day, preds=preds, y_pred=pred_z_scores, y_true=true_z_scores)
                positions = weights * total_portfolio_value
                

        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)

        self._plotter.plot_equity_cruve(title, daily_res, self._snp500_daily_returns)
        plot_type = 2 if model_type is None or model_type == "random" else 3
        self._plotter.plot_metrics(title, plot_type, monthly_returns, self._rfr, daily_portfolio_values, total_portfolio_value, monthly_portfolio_values, self._backtest_months, self._fama, pred_returns, true_returns, pred_z_scores, true_z_scores)
        self._plotter.plot_dd(title, daily_portfolio_values, daily_res)

        self._monte_carlo.monte_carlo_path_sim(title, daily_res, self._rfr, self._snp500_cagr, self._backtest_months)