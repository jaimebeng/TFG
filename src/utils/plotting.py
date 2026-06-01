import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import os
import numpy as np
from src.utils.metrics import *
from matplotlib import colors as mcolors


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
        path = os.path.join(self._output_path, title.replace(' ', '_'))
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f"{title.replace(' ', '_')}_backtest.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_metrics(self, title, plot_type, monthly_returns, rfr, daily_portfolio_values, total_portfolio_value, monthly_portfolio_values, backtest_months, fama = None, total_trans_costs = None, pred_returns = None, true_returns = None, pred_z_scores = None, true_z_scores = None):

        portfolio_volatility_value = portfolio_volatility(monthly_returns)
        sharpe_ratio_value = sharpe_ratio(monthly_returns, rfr)
        sortino_ratio_value = sortino_ratio(monthly_returns, rfr)
        green_ratio, red_ratio = green_red_months(monthly_returns)
        avg_green, avg_red = monthly_averages(monthly_returns)
        dds = drawdowns(daily_portfolio_values)
        max_cap_loss = max_capital_loss(daily_portfolio_values)
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
            ("Max Capital Loss", f"{max_cap_loss:.2%}"),
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
            metrics.insert(1, ("Total Transaction Costs", f"{total_trans_costs:.2%}"))
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
        fig_table, ax_table = plt.subplots(figsize=(10, 6.8), constrained_layout=True)
        ax_table.axis("off")
        table = ax_table.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.35, 0.2],
            bbox=[0, 0, 1, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.28)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D7DE")
            if row == 0:
                cell.set_facecolor("#0B5CAD")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("white")
        fig_table.suptitle(f"{title} Metrics", fontsize=14, fontweight="bold", y=0.98)
        path = os.path.join(self._output_path, title.replace(' ', '_'))
        os.makedirs(path, exist_ok=True)
        fig_table.savefig(os.path.join(path, f"{title.replace(' ', '_')}_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_table)

    def plot_dd(self, title, daily_portfolio_values, daily_res):
        dds = drawdowns(daily_portfolio_values)
        drawdown_pct = dds.to_numpy() * 100
        drawdown_pct = pd.Series(drawdown_pct[1:], index=daily_res.index)
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
        path = os.path.join(self._output_path, title.replace(' ', '_'))
        os.makedirs(path, exist_ok=True)
        fig_dd.savefig(os.path.join(path, f"{title.replace(' ', '_')}_dd.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_dd)

    def plot_mc_metrics(self, title, total_return_values, portfolio_volatility_values, ulcer_index_values, average_dd_duration_values, cagr_values, snp500_cagr, sharpe_ratio_values, max_dd_values, final_portfolio_values):

        total_return_mean, total_return_median = total_return_stats(total_return_values)
        min_return, stress, bad, typical, good, upside, max_return = total_return_quantiles(total_return_values)
        mean_volatility = mean_port_volatility(portfolio_volatility_values)
        mean_ulcer_index_value = mean_ulcer_index(ulcer_index_values)
        mean_avg_dd_duration_value = mean_avg_dd_duration(average_dd_duration_values)
        prob_cagr_over_null_value = prob_cagr_over_null(cagr_values)
        prob_cagr_over_benchmark_value = prob_cagr_over_benchmark(cagr_values, snp500_cagr)
        prob_sharpe_over_1_value = prob_sharpe_over_1(sharpe_ratio_values)
        prob_max_dd_over_20 = prob_max_dd_over(max_dd_values, -0.2)
        prob_max_dd_over_50 = prob_max_dd_over(max_dd_values, -0.5)
        prob_avg_dd_duration_over_365_value = prob_avg_dd_duration_over_365(average_dd_duration_values)
        prob_negative_return_value = prob_negative_return(final_portfolio_values)
        prob_loss_over_20 = prob_loss_over(final_portfolio_values, 0.2)
        prob_loss_over_50 = prob_loss_over(final_portfolio_values, 0.5)

        summary_metrics = pd.DataFrame(
            [
                ("Total Return Mean", f"{total_return_mean:.2%}"),
                ("Total Return Median", f"{total_return_median:.2%}"),
                ("Min Total Return Value", f"{min_return:.2%}"),
                ("0.5% Percentile", f"{stress:.2%}"),
                ("5% Percentile", f"{bad:.2%}"),
                ("50% Percentile", f"{typical:.2%}"),
                ("95% Percentile", f"{good:.2%}"),
                ("99.5% Percentile", f"{upside:.2%}"),
                ("Max Total Return Value", f"{max_return:.2%}"),
                ("Mean Monthly Volatility", f"{mean_volatility:.2%}"),
                ("Mean Ulcer Index", f"{mean_ulcer_index_value:.4f}"),
                ("Mean Avg DD Duration", f"{mean_avg_dd_duration_value:.2f} days"),
                ("P(CAGR > 0)", f"{prob_cagr_over_null_value:.2%}"),
                ("P(CAGR > S&P 500 CAGR)", f"{prob_cagr_over_benchmark_value:.2%}"),
                ("P(Sharpe > 1)", f"{prob_sharpe_over_1_value:.2%}"),
                ("P(Max DD < -20%)", f"{prob_max_dd_over_20:.2%}"),
                ("P(Max DD < -50%)", f"{prob_max_dd_over_50:.2%}"),
                ("P(Avg DD Duration > 365)", f"{prob_avg_dd_duration_over_365_value:.2%}"),
                ("P(Negative Return)", f"{prob_negative_return_value:.2%}"),
                ("P(Loss > 20%)", f"{prob_loss_over_20:.2%}"),
                ("P(Loss > 50%)", f"{prob_loss_over_50:.2%}"),
            ],
            columns=["Metric", "Value"],
        )

        fig_table, ax_table = plt.subplots(figsize=(10, 8.0), constrained_layout=True)
        ax_table.axis("off")
        table = ax_table.table(
            cellText=summary_metrics.values,
            colLabels=summary_metrics.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.62, 0.22],
            bbox=[0, 0, 1, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.24)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D7DE")
            if row == 0:
                cell.set_facecolor("#0B5CAD")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("white")

        fig_table.suptitle(f"{title} Monte Carlo Metrics", fontsize=14, fontweight="bold", y=0.98)
        path = os.path.join(self._output_path, title.replace(' ', '_'))
        os.makedirs(path, exist_ok=True)
        fig_table.savefig(os.path.join(path, f"{title.replace(' ', '_')}_mc_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_table)

    def plot_mc_histograms(self, plot_title, total_return_values, sharpe_ratio_values, max_dd_values, cvar_values, average_dd_duration_values):

        histogram_data = [
            ("Total Return Value", total_return_values, "#0B5CAD"),
            ("Sharpe Ratio", sharpe_ratio_values, "#2EAD4A"),
            ("Max Drawdown", max_dd_values, "#D64541"),
            ("CVaR", cvar_values, "#7A4CC2"),
            ("Max DD Duration", average_dd_duration_values, "#E67E22"),
        ]

        fig_hist, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        axes = axes.ravel()

        for index, (title, values, color) in enumerate(histogram_data):
            ax = axes[index]
            vals = np.array(values)
            ax.hist(vals, bins=50, color=color, alpha=0.78, edgecolor="white")
            ax.axvline(np.mean(vals), color="#1F2937", linestyle="--", linewidth=1.5, label="Mean")
            ax.set_title(title, fontweight="bold")
            ax.set_xlim(vals.min(), vals.max())

            ax.grid(True, alpha=0.2)
            ax.legend(frameon=True)

        axes[-1].axis("off")
        
        fig_hist.suptitle(f"{plot_title} Monte Carlo Histograms", fontsize=16, fontweight="bold")
        path = os.path.join(self._output_path, plot_title.replace(' ', '_'))
        os.makedirs(path, exist_ok=True)
        fig_hist.savefig(os.path.join(path, f"{plot_title.replace(' ', '_')}_mc_histograms.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_hist)

    def plot_mc_paths(self, title, daily_portfolios_values, backtest_months):
        portfolio_paths = daily_portfolios_values
        portfolio_paths = np.concatenate([np.ones((portfolio_paths.shape[0], 1)), portfolio_paths], axis=1)
        portfolio_paths_pct = portfolio_paths * 100
        portfolio_paths_pct[:, 0] = 100

        fig_paths, ax_paths = plt.subplots(figsize=(16, 8), constrained_layout=True)
        x = np.arange(portfolio_paths_pct.shape[1])

        n_paths = portfolio_paths_pct.shape[0]
        base = plt.cm.turbo(np.linspace(0, 1, n_paths))
        np.random.shuffle(base)
        path_colors = [mcolors.to_hex(c) for c in base]

        for idx, path in enumerate(portfolio_paths_pct):
            ax_paths.plot(x, path, color=path_colors[idx], alpha=0.2, linewidth=0.9, zorder=1)

        median_path = np.median(portfolio_paths_pct, axis=0)
        ax_paths.plot(x, median_path, color="#08306B", linewidth=3.6, label="Median", zorder=3)

        monthly_labels = [d.strftime("%Y-%m") for d in backtest_months]
        monthly_positions = np.linspace(0, portfolio_paths_pct.shape[1]-1, num=len(monthly_labels), dtype=int)
        if len(monthly_labels) > 12:
            step = int(np.ceil(len(monthly_labels)/12))
            monthly_labels = monthly_labels[::step]
            monthly_positions = monthly_positions[::step]

        ax_paths.set_xticks(monthly_positions)
        ax_paths.set_xticklabels(monthly_labels, rotation=45)

        ax_paths.set_title(f"{title} Monte Carlo Paths", fontsize=16, fontweight="bold")
        ax_paths.set_xlabel("Date (monthly)")
        ax_paths.set_ylabel("Portfolio Value (% of initial)")
        ax_paths.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        all_min = portfolio_paths_pct.min()
        all_max = portfolio_paths_pct.max()
        pad = max(1.0, (all_max - all_min) * 0.03)
        ax_paths.set_ylim(bottom=all_min - pad, top=all_max + pad)
        ax_paths.grid(True, alpha=0.18)
        ax_paths.legend(frameon=True)
        path = os.path.join(self._output_path, title.replace(' ', '_'))
        os.makedirs(path, exist_ok=True)
        fig_paths.savefig(os.path.join(path, f"{title.replace(' ', '_')}_mc_paths.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_paths)

