import numpy as np
from scipy.stats.mstats import spearmanr
import pandas as pd
import math
import statsmodels.api as sm
import math
from scipy.stats.mstats import spearmanr
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

def bids(y_pred, y_true):

    rank_ic, _ = spearmanr(y_true,y_pred)
    if np.isnan(rank_ic):
        rank_ic = 0

    directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
    da_centered = 2 * (directional_accuracy - 0.5)

    return 0.6 * rank_ic + 0.4 * da_centered

def portfolio_volatility(monthly_returns):

    return np.std(monthly_returns, ddof=1)

def sharpe_ratio(monthly_returns, risk_free_rate):
    excess = np.array(monthly_returns) - np.array(risk_free_rate)
    mean_excess = np.mean(excess)
    vol_excess = np.std(excess, ddof=1)
    monthly_sharpe = (mean_excess) / vol_excess
    annualized_sharpe = monthly_sharpe * np.sqrt(12)

    return annualized_sharpe

def sortino_ratio(monthly_returns, risk_free_rate):
    excess = np.array(monthly_returns) - np.array(risk_free_rate)
    mean_excess = np.mean(excess)
    negative_excess = np.minimum(excess, 0)
    downside_dev = np.sqrt(np.mean(negative_excess ** 2))

    annualized_sortino = (mean_excess / downside_dev) * np.sqrt(12)

    return annualized_sortino

def green_red_months(monthly_returns):
    monthly_returns = np.array(monthly_returns)
    green_ratio = np.mean(monthly_returns > 0) 
    red_ratio = np.mean(monthly_returns <= 0)

    return green_ratio, red_ratio

def monthly_averages(monthly_returns):
    monthly_returns = np.array(monthly_returns)
    green_avg = np.mean(monthly_returns[monthly_returns > 0])
    red_avg = np.mean(monthly_returns[monthly_returns <= 0])

    return green_avg, red_avg

def max_capital_loss(portfolio_values):
    
    return min(0.0, np.min(portfolio_values) - 1.0)

def drawdowns(portfolio_values):
    portfolio_values = pd.Series(portfolio_values)
    portfolio_values = pd.concat([pd.Series([1]), portfolio_values], ignore_index=True)
    running_peak = portfolio_values.cummax()
    drawdowns = (portfolio_values / running_peak) - 1

    return drawdowns

def max_drawdown(drawdowns):
    
    return np.min(drawdowns)

def average_drawdown_depth(drawdowns):
    drawdowns = pd.Series(drawdowns)
    in_dd = drawdowns < 0
    groups = (in_dd != in_dd.shift()).cumsum()
    troughs = drawdowns.groupby(groups).min()
    troughs = troughs[troughs < 0]

    if len(troughs) == 0:
        return 0.0

    return troughs.mean()

def max_drawdown_duration(portfolio_values):
    portfolio_values = pd.Series(portfolio_values)
    in_dd = portfolio_values < portfolio_values.cummax()
    groups = (in_dd != in_dd.shift()).cumsum()
    durations = in_dd.groupby(groups).sum()

    if len(durations) == 0:
        return 0

    return durations.max()

def average_drawdown_duration(portfolio_values):
    portfolio_values = pd.Series(portfolio_values)
    in_dd = portfolio_values < portfolio_values.cummax()
    groups = (in_dd != in_dd.shift()).cumsum()
    durations = in_dd.groupby(groups).sum()
    durations = durations[durations > 0]

    if len(durations) == 0:
        return 0.0

    return durations.mean()

def ulcer_index(drawdowns):

    return np.sqrt(np.mean(np.power(drawdowns,2)))

def var(monthly_returns, alpha = 0.05):
    monthly_returns = np.array(monthly_returns)
    var = np.quantile(monthly_returns, alpha)

    return var

def cvar(monthly_returns, alpha = 0.05):
    monthly_returns = np.array(monthly_returns)
    var = np.quantile(monthly_returns, alpha)
    cvar = monthly_returns[monthly_returns <= var].mean()

    return cvar

def cagr(ending_value, years):
    return np.power((ending_value / 1), (1 / years)) - 1

def t_stat(monthly_returns):
    T = len(monthly_returns)
    L = math.ceil(4 * np.power((T / 100), 2 / 9))
    X = np.ones((T, 1))    
    model = sm.OLS(monthly_returns, X)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags' : L})
    t_stat = res.tvalues[0]
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    return t_stat, p_value

def fama_regression(monthly_returns, fama):
    y = np.array(monthly_returns) - np.array(fama["RF"])
    X = fama[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    T = len(monthly_returns)
    L = math.ceil(4 * np.power((T / 100), 2 / 9))
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": L})

    alpha = res.params["const"]
    betas = res.params[['Mkt-RF','SMB','HML','RMW','CMA']]
    alpha_p_value = res.pvalues["const"]
    annualized_alpha = 12 * alpha
    
    return annualized_alpha, alpha_p_value, betas

def bl_ER_precision(preds, true):
    ics = [spearmanr(y_pred,y_true)[0] for y_pred,y_true in zip(preds,true)]
    mses = [mean_squared_error(y_pred,y_true) for y_pred,y_true in zip(preds,true)]
    norm_mse = np.mean(mses) / np.var(np.vstack(true))
    das = [np.mean(np.sign(y_pred) == np.sign(y_true)) for y_pred,y_true in zip(preds,true)]

    bl_mat = np.vstack([x.ravel() for x in preds])
    true_mat = np.vstack([x.ravel() for x in true])
    model_mse = np.mean((bl_mat[1:] - true_mat[1:])**2)
    rw_pred = true_mat[:-1]
    rw_actual = true_mat[1:]
    rw_mse = np.mean((rw_pred - rw_actual)**2)
    u2 = np.sqrt(model_mse / rw_mse)
    
    return np.mean(ics), norm_mse, np.mean(mses), np.mean(das), u2

def model_precision(preds, true):
    ics = [spearmanr(y_pred,y_true)[0] for y_pred,y_true in zip(preds,true)]
    mses = [mean_squared_error(y_pred,y_true) for y_pred,y_true in zip(preds,true)]
    da = np.mean(np.sign(preds) == np.sign(true))

    return np.mean(ics), np.mean(mses), da

def calculate_final_portfolio(monthly_returns):
    final_value = 1 * np.prod(1 + monthly_returns)

    return final_value

def calculate_daily_portfolio_values(monthly_returns):
    daily_portfolio_values = np.cumprod(1 + monthly_returns)

    return daily_portfolio_values

def total_return_stats(total_return_values):

    return np.mean(total_return_values), np.median(total_return_values)

def total_return_quantiles(total_return_values):
    total_return_values = np.array(total_return_values)

    min_return = np.min(total_return_values)
    stress = np.quantile(total_return_values, 0.005)
    bad = np.quantile(total_return_values, 0.05)
    typical = np.quantile(total_return_values, 0.5)
    good = np.quantile(total_return_values, 0.95)
    upside = np.quantile(total_return_values, 0.995)
    max_return = np.max(total_return_values)

    return min_return, stress, bad, typical, good, upside, max_return

def mean_port_volatility(portfolio_volatility_values):

    return np.mean(portfolio_volatility_values)

def mean_ulcer_index(ulcer_index_values):

    return np.mean(ulcer_index_values)

def mean_avg_dd_duration(avg_dd_duration_values):

    return np.mean(avg_dd_duration_values)

def prob_cagr_over_null(cagr_values):
    cagr_values = np.array(cagr_values)

    return len(cagr_values[cagr_values > 0]) / len(cagr_values)

def prob_cagr_over_benchmark(cagr_values, benchmark_cagr_value):
    cagr_values = np.array(cagr_values)

    return len(cagr_values[cagr_values > benchmark_cagr_value]) / len(cagr_values)

def prob_sharpe_over_1(sharpe_values):
    sharpe_values = np.array(sharpe_values)

    return len(sharpe_values[sharpe_values > 1]) / len(sharpe_values)

def prob_max_dd_over(max_dd_values, threshold):
    max_dd_values = np.array(max_dd_values)

    return len(max_dd_values[max_dd_values < threshold]) / len(max_dd_values)

def prob_avg_dd_duration_over_365(avg_dd_duration_values):
    avg_dd_duration_values = np.array(avg_dd_duration_values)

    return len(avg_dd_duration_values[avg_dd_duration_values > 365]) / len(avg_dd_duration_values)

def prob_negative_return(final_portfolio_values):
    final_portfolio_values = np.array(final_portfolio_values)

    return len(final_portfolio_values[final_portfolio_values < 1]) / len(final_portfolio_values)

def prob_loss_over(final_portfolio_values, threshold):
    final_portfolio_values = np.array(final_portfolio_values)
    loss = 1 - final_portfolio_values

    return len(loss[loss > threshold]) / len(loss)