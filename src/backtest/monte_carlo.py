import os
import sys
sys.path.append(os.path.abspath(".."))
from src.utils.metrics import *
from src.utils.plotting import Plotter
import numpy as np
import pandas as pd

np.random.seed(42)

class MonteCarlo():

    def __init__(self):
        self._plotter = Plotter()

    def monte_carlo_path_sim(self, title, daily_res, rfr, snp500_cagr, backtest_months):
        df = daily_res.copy()
        r = df["Returns"].to_numpy()
        n = len(r)
        L = 21
        p = 1 / L

        final_portfolio_values = np.empty((10000))
        total_return_values = np.empty((10000))
        portfolio_volatility_values = np.empty((10000))
        sharpe_ratio_values = np.empty((10000))
        max_dd_values = np.empty((10000))
        average_dd_duration_values = np.empty((10000))
        ulcer_index_values = np.empty((10000))
        cvar_values = np.empty((10000))
        cagr_values = np.empty((10000))
        daily_portfolios_values = np.empty((10000, n))

        for i in range(10000):
            sample = np.empty(n)
            j = np.random.randint(n)

            for t in range(n):
                if np.random.rand() < p:
                    j = np.random.randint(n)
                sample[t] = r[j]
                j = (j + 1) % n

            daily_returns = sample
            final_portfolio_value = calculate_final_portfolio(daily_returns)
            final_portfolio_values[i] = final_portfolio_value
            total_return_values[i] = final_portfolio_value - 1
            daily_portfolio_values = calculate_daily_portfolio_values(daily_returns)
            daily_portfolios_values[i] = daily_portfolio_values
            daily_returns = pd.DataFrame({"Returns" : daily_returns}, index=df.index)
            monthly_res = (1 + daily_returns["Returns"]).resample("ME").prod() - 1
            monthly_returns = monthly_res.to_numpy()
            portfolio_volatility_value = portfolio_volatility(monthly_returns)
            portfolio_volatility_values[i] = portfolio_volatility_value
            sharpe_ratio_value = sharpe_ratio(monthly_returns, rfr)
            sharpe_ratio_values[i] = sharpe_ratio_value
            dds = drawdowns(daily_portfolio_values)
            max_dd = max_drawdown(dds)
            max_dd_values[i] = max_dd
            average_dd_duration = average_drawdown_duration(daily_portfolio_values)
            average_dd_duration_values[i] = average_dd_duration
            ulcer_index_value = ulcer_index(dds)
            ulcer_index_values[i] = ulcer_index_value
            cvar_value = cvar(monthly_returns)
            cvar_values[i] = cvar_value
            cagr_value = cagr(final_portfolio_value, len(monthly_returns)/12)
            cagr_values[i] = cagr_value
        

        self._plotter.plot_mc_metrics(title, total_return_values, portfolio_volatility_values, ulcer_index_values, average_dd_duration_values, cagr_values, snp500_cagr, sharpe_ratio_values, max_dd_values, final_portfolio_values)
        self._plotter.plot_mc_histograms(title, total_return_values, sharpe_ratio_values, max_dd_values, cvar_values, average_dd_duration_values)
        self._plotter.plot_mc_paths(title, daily_portfolios_values, backtest_months)


    