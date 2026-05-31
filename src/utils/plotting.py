import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import os
import numpy as np
from src.utils.metrics import *

plt.style.use("seaborn-v0_8-whitegrid")

class Plotter():
    def __init__(self):
        self._output_path = "/home/jaime/Documents/TFG/results"
        os.makedirs(self._output_path, exist_ok=True)


    def plot_equity_cruve(self, title, daily_res, snp500_daily_returns):

        daily_pct = daily_res["Portfolio Growth"].to_numpy() * 100
        daily_pct = pd.Series(daily_pct, index=daily_res.index)
        snp500_pct = pd.Series(np.cumprod(1 + np.asarray(snp500_daily_returns)) * 100, index=daily_res.index)
        daily_pct.iloc[0] = 100
        snp500_pct.iloc[0] = 100

        fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
        positive_color = "#2EAD4A"
        negative_color = "#D64541"
        portfolio_returns = daily_res["Returns"].to_numpy()

        ax.plot([], [], color=positive_color, linewidth=2.7, label="Positive day")
        ax.plot([], [], color=negative_color, linewidth=2.7, label="Negative day")

        for i in range(1, len(daily_pct.index)):
            segment_color = positive_color if portfolio_returns[i] >= 0 else negative_color
            ax.plot(
                daily_pct.index[i - 1:i + 1],
                daily_pct.values[i - 1:i + 1],
                color=segment_color,
                linewidth=2.7,
            )

        ax.set_title(f"{title} Backtest Performance", fontsize=20, fontweight="bold", pad=16)
        ax.set_ylabel("Portfolio value (% of initial capital)", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(True, which="major", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=True)

        fig.savefig(os.path.join(self._output_path, f"{title.replace(' ', '_')}_backtest.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_metrics(self, title, plot_type, monthly_returns, rfr, daily_portfolio_values, total_portfolio_value, monthly_portfolio_values, backtest_months, fama = None, pred_returns = None, true_returns = None, pred_z_scores = None, true_z_scores = None):

        portfolio_volatility_value = portfolio_volatility(monthly_returns)
        sharpe_ratio_value = sharpe_ratio(monthly_returns, rfr)
        sortino_ratio_value = sortino_ratio(monthly_returns, rfr)
        green_ratio, red_ratio = green_red_months(monthly_returns)
        avg_green, avg_red = monthly_averages(monthly_returns)
        dds = drawdowns(daily_portfolio_values)
        max_dd = max_drawdown(dds)
        average_dd_depth = average_drawdown_depth(dds)
        max_dd_duration = max_drawdown_duration(daily_portfolio_values)
        average_dd_duration = average_drawdown_duration(daily_portfolio_values)
        ulcer_index_value = ulcer_index(dds)
        var_value = var(monthly_returns)
        cvar_value = cvar(monthly_returns)
        cagr_value = cagr(monthly_portfolio_values[-1], len(backtest_months)/12)
        t_statistic, t_p_value = t_stat(monthly_returns)

        if plot_type > 0:
            alpha, a_p_value, betas = fama_regression(monthly_returns, fama)
        if plot_type > 1:
            bl_IC, bl_MSE_norm, bl_MSE_raw, bl_DA, bl_U2 = bl_ER_precision(pred_returns, true_returns)
        if plot_type > 2:
            model_IC, model_MSE, model_DA = model_precision(pred_z_scores, true_z_scores)

        metrics = [
            ("Total Return", f"{total_portfolio_value - 1:.2%}"),
            ("Monthly Volatility", f"{portfolio_volatility_value:.2%}"),
            ("Sharpe ratio", f"{sharpe_ratio_value:.2f}"),
            ("Sortino ratio", f"{sortino_ratio_value:.2f}"),
            ("Positive Months", f"{green_ratio:.2%}"),
            ("Negative Months", f"{red_ratio:.2%}"),
            ("Avg Positive Month", f"{avg_green:.2%}"),
            ("Avg Negative Month", f"{avg_red:.2%}"),
            ("Max drawdown", f"{max_dd:.2%}"),
            ("Avg drawdown", f"{average_dd_depth:.2%}"),
            ("Max DD duration", f"{max_dd_duration:.0f} days"),
            ("Avg DD duration", f"{average_dd_duration:.2f} days"),
            ("Ulcer Index", f"{ulcer_index_value:.2f}"),
            ("VaR (5%)", f"{var_value:.2%}"),
            ("CVaR (5%)", f"{cvar_value:.2%}"),
            ("CAGR", f"{cagr_value:.2%}"),
            ("t-stat", f"{t_statistic:.2f}"),
            ("t-stat p-value", f"{t_p_value:.4f}"),
        ]

        if plot_type > 0:
            metrics.append(("Market beta", f"{betas['Mkt-RF']:.2f}"))
            metrics.append(("Size beta", f"{betas['SMB']:.2f}"))
            metrics.append(("Value beta", f"{betas['HML']:.2f}"))
            metrics.append(("Profitability beta", f"{betas['RMW']:.2f}"))
            metrics.append(("Investment beta", f"{betas['CMA']:.2f}"))
            metrics.append(("Alpha", f"{alpha:.2f}"))
            metrics.append(("Alpha p-value", f"{a_p_value:.4f}"))
        if plot_type > 1:
            metrics.append(("Black Litterman IC", f"{bl_IC:.2f}"))
            metrics.append(("Black Litterman Normalized MSE", f"{bl_MSE_norm:.2f} ({bl_MSE_raw:.4f})"))
            metrics.append(("Black Litterman DA", f"{bl_DA:.2f}"))
            metrics.append(("Black Litterman Thiel's U2", f"{bl_U2:.2f}"))
        if plot_type > 2:
            metrics.append(("Model IC", f"{model_IC:.2f}"))
            metrics.append(("Model MSE", f"{model_MSE:.2f}"))
            metrics.append(("Model DA", f"{model_DA:.2f}"))

        metrics_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
        fig_table, ax_table = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
        ax_table.axis("off")
        table = ax_table.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.35, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.2)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D7DE")
            if row == 0:
                cell.set_facecolor("#0B5CAD")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("white")
        ax_table.set_title(f"{title} Metrics", fontsize=14, fontweight="bold", pad=12)

        fig_table.savefig(os.path.join(self._output_path, f"{title.replace(' ', '_')}_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_table)

    def plot_dd(self, title, daily_portfolio_values, daily_res):
        dds = drawdowns(daily_portfolio_values)
        drawdown_pct = dds.to_numpy() * 100
        drawdown_pct = pd.Series(drawdown_pct, index=daily_res.index)
        drawdown_pct.iloc[0] = 0

        fig_dd, ax_dd = plt.subplots(figsize=(16, 4.8), constrained_layout=True)
        ax_dd.plot(
            drawdown_pct.index,
            drawdown_pct.values,
            color="#C0392B",
            linewidth=2.2,
            label="Drawdown",
        )
        ax_dd.axhline(0, color="#2F3B52", linewidth=1.1, linestyle="--", alpha=0.8)
        ax_dd.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="#C0392B", alpha=0.14)
        ax_dd.set_title(f"{title} Drawdown", fontsize=18, fontweight="bold", pad=14)
        ax_dd.set_ylabel("Drawdown (%)", fontsize=12)
        ax_dd.set_xlabel("Date", fontsize=12)
        ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax_dd.grid(True, which="major", alpha=0.25)
        ax_dd.spines["top"].set_visible(False)
        ax_dd.spines["right"].set_visible(False)
        ax_dd.legend(loc="lower left", frameon=True)

        fig_dd.savefig(os.path.join(self._output_path, f"{title.replace(' ', '_')}_dd.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_dd)