from src.data.data_loader import DataLoad
from src.utils.plotting import Plotter
from src.models.create_shallow_models import lin_reg
from src.models.create_hpt_search import *
from src.models.create_pytorch_hpt_search import *
from src.portfolio.bl_optimization import BlackLitterman
from src.backtest.monte_carlo import MonteCarlo
from src.utils.metrics import cagr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.pytorch_optuna_search import train_model
from src.models.create_nn_models import MODEL_REGISTRY
from torch.utils.data import DataLoader, TensorDataset


TRANSACTION_COSTS = 0.0005
SLIPPAGE = 0.0005

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

    def _calculate_turnover(self, new_weights, old_weights):
        turnover = 0.5 * np.sum(np.abs(np.array(new_weights) - np.array(old_weights)))

        return turnover
        
    def eq_portfolio_backtest(self):
        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        net_total_portfolio_value = 1
        gross_total_portfolio_value = 1
        monthly_start_value = 1
        weights = np.ones(30) / 30
        turnover = self._calculate_turnover(weights, np.zeros(30))
        trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
        net_total_portfolio_value -= trans_costs
        net_positions = weights * net_total_portfolio_value
        gross_positions = weights * gross_total_portfolio_value

        for day in self._backtest_days:
            
            r = self._daily_stock_returns.loc[day].values 
            net_pnl = np.sum(net_positions * r)
            net_portfolio_return = net_pnl / net_total_portfolio_value
            daily_returns.append(net_portfolio_return)
            net_total_portfolio_value += net_pnl
            daily_portfolio_values.append(net_total_portfolio_value)
            net_positions = net_positions * (1 + r)
            weights = net_positions / net_total_portfolio_value
            gross_pnl = np.sum(gross_positions * r)
            gross_total_portfolio_value += gross_pnl
            gross_positions = gross_positions * (1 + r)
            

            if day in self._backtest_months:
                monthly_return = (net_total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(net_total_portfolio_value)
                monthly_start_value = net_total_portfolio_value
                old_weights = weights
                weights = np.ones(30) / 30
                turnover = self._calculate_turnover(weights, old_weights)
                trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
                net_total_portfolio_value -= trans_costs
                net_positions = weights * net_total_portfolio_value
                gross_positions = weights * gross_total_portfolio_value


        total_trans_costs = gross_total_portfolio_value - net_total_portfolio_value
        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)
        
        self._plotter.plot_equity_cruve("Equally Weighted Portfolio", daily_res, self._snp500_daily_returns)
        self._plotter.plot_metrics("Equally Weighted Portfolio", 1, monthly_returns, self._rfr, daily_portfolio_values, net_total_portfolio_value, monthly_portfolio_values, self._backtest_months, fama=self._fama, total_trans_costs=total_trans_costs)
        self._plotter.plot_dd("Equally Weighted Portfolio", daily_portfolio_values, daily_res)

        self._monte_carlo.monte_carlo_path_sim("Equally Weighted Portfolio", daily_res, self._rfr, self._snp500_cagr, self._backtest_months)

    def mvo_portfolio_backtest(self):
        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        net_total_portfolio_value = 1
        gross_total_portfolio_value = 1
        monthly_start_value = 1
        bl = BlackLitterman()
        first_day = self._backtest[self._backtest.index <= self._backtest_days[0]].index[-1]
        preds = pd.DataFrame({"Ticker": self._backtest.loc[first_day]["Ticker"]})
        preds["Predictions"] = self._monthly_stock_returns.loc[first_day].values
        weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= first_day].tail(504), preds=preds)
        turnover = self._calculate_turnover(weights, np.zeros(30))
        trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
        net_total_portfolio_value -= trans_costs
        net_positions = weights * net_total_portfolio_value
        gross_positions = weights * gross_total_portfolio_value
        pred_returns = []
        true_returns = []

        for day in self._backtest_days:

            r = self._daily_stock_returns.loc[day].values 
            net_pnl = np.sum(net_positions * r)
            net_portfolio_return = net_pnl / net_total_portfolio_value
            daily_returns.append(net_portfolio_return)
            net_total_portfolio_value += net_pnl
            daily_portfolio_values.append(net_total_portfolio_value)
            net_positions = net_positions * (1 + r)
            weights = net_positions / net_total_portfolio_value
            gross_pnl = np.sum(gross_positions * r)
            gross_total_portfolio_value += gross_pnl
            gross_positions = gross_positions * (1 + r)

            if day in self._backtest_months:
                monthly_return = (net_total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(net_total_portfolio_value)
                monthly_start_value = net_total_portfolio_value
                pred_returns.append(expected_returns)
                true_returns.append(self._monthly_stock_returns.loc[day].values)
                preds["Predictions"] = self._monthly_stock_returns.loc[day].values
                old_weights = weights
                weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= day].tail(504), preds=preds)
                turnover = self._calculate_turnover(weights, old_weights)
                trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
                net_total_portfolio_value -= trans_costs
                net_positions = weights * net_total_portfolio_value
                gross_positions = weights * gross_total_portfolio_value

        total_trans_costs = gross_total_portfolio_value - net_total_portfolio_value
        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)

        self._plotter.plot_equity_cruve("MVO Portfolio", daily_res, self._snp500_daily_returns)
        self._plotter.plot_metrics("MVO Portfolio", 1, monthly_returns, self._rfr, daily_portfolio_values, net_total_portfolio_value, monthly_portfolio_values, self._backtest_months, fama=self._fama, total_trans_costs=total_trans_costs)
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
        net_total_portfolio_value = 1
        gross_total_portfolio_value = 1
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
        turnover = self._calculate_turnover(weights, np.zeros(30))
        trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
        net_total_portfolio_value -= trans_costs
        net_positions = weights * net_total_portfolio_value
        gross_positions = weights * gross_total_portfolio_value
        pred_returns = []
        true_returns = []

        for day in self._backtest_days:

            r = self._daily_stock_returns.loc[day].values 
            net_pnl = np.sum(net_positions * r)
            net_portfolio_return = net_pnl / net_total_portfolio_value
            daily_returns.append(net_portfolio_return)
            net_total_portfolio_value += net_pnl
            daily_portfolio_values.append(net_total_portfolio_value)
            net_positions = net_positions * (1 + r)
            weights = net_positions / net_total_portfolio_value
            gross_pnl = np.sum(gross_positions * r)
            gross_total_portfolio_value += gross_pnl
            gross_positions = gross_positions * (1 + r)

            if day in self._backtest_months:
                monthly_return = (net_total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(net_total_portfolio_value)
                monthly_start_value = net_total_portfolio_value
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
                old_weights = weights
                weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= day].tail(504), day, preds=preds, y_pred=pred_z_scores, y_true=true_z_scores)
                turnover = self._calculate_turnover(weights, old_weights)
                trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
                net_total_portfolio_value -= trans_costs
                net_positions = weights * net_total_portfolio_value
                gross_positions = weights * gross_total_portfolio_value
                
        total_trans_costs = gross_total_portfolio_value - net_total_portfolio_value
        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)

        self._plotter.plot_equity_cruve(title, daily_res, self._snp500_daily_returns)
        plot_type = 2 if model_type == 0 or model_type == "random" else 3
        self._plotter.plot_metrics(title, plot_type, monthly_returns, self._rfr, daily_portfolio_values, net_total_portfolio_value, monthly_portfolio_values, self._backtest_months, self._fama, total_trans_costs, pred_returns, true_returns, pred_z_scores, true_z_scores)
        self._plotter.plot_dd(title, daily_portfolio_values, daily_res)

        self._monte_carlo.monte_carlo_path_sim(title, daily_res, self._rfr, self._snp500_cagr, self._backtest_months)

    def pytorch_portfolio_backtest(self, title, model_type, device = None):
        daily_returns = []
        monthly_returns = []
        daily_portfolio_values = []
        monthly_portfolio_values = []
        net_total_portfolio_value = 1
        gross_total_portfolio_value = 1
        monthly_start_value = 1
        preds = None
        pred_z_scores = None
        true_z_scores = None
        features = [c for c in self._backtest.columns if c not in ["Ticker", "Target"]]
        input_features = len(features)
        grid = create_pytorch_search(model_type, device=device, input_features=input_features)
        prev_day = self._first_day - pd.Timedelta(days=1)
        X, y = self._dl.load_dataset("hpt", prev_day)
        grid.fit(X,y)
        model = grid.best_model_
        best_params = grid.best_params_
        model.eval()
        with torch.no_grad():
            if model_type == "mlp":
                df_first = self._backtest.loc[[self._first_day]].sort_values(by="Ticker")
                mlp_in = torch.from_numpy(df_first[features].values).float().to(device)
                pred = model(mlp_in)
            else:
                pred_input = self._prepare_sequential_data(self._first_day, features, device, is_inference=True)
                pred = model(pred_input)
        
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        preds = pd.DataFrame({"Ticker": self._backtest.loc[self._first_day]["Ticker"]}).sort_values(by="Ticker").reset_index(drop=True)
        df_first_sorted = self._backtest.loc[[self._first_day]].sort_values(by="Ticker")
        preds["Predictions"] = pred
        pred_z_scores = [pred]
        true_z_scores = [df_first_sorted["Target"].to_numpy()]
        bl = BlackLitterman()
        weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= self._first_day].tail(504), self._first_day, preds=preds)
        turnover = self._calculate_turnover(weights, np.zeros(30))
        trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
        net_total_portfolio_value -= trans_costs
        net_positions = weights * net_total_portfolio_value
        gross_positions = weights * gross_total_portfolio_value
        pred_returns = []
        true_returns = []

        for day in self._backtest_days:

            r = self._daily_stock_returns.loc[day].values 
            net_pnl = np.sum(net_positions * r)
            net_portfolio_return = net_pnl / net_total_portfolio_value
            daily_returns.append(net_portfolio_return)
            net_total_portfolio_value += net_pnl
            daily_portfolio_values.append(net_total_portfolio_value)
            net_positions = net_positions * (1 + r)
            weights = net_positions / net_total_portfolio_value
            gross_pnl = np.sum(gross_positions * r)
            gross_total_portfolio_value += gross_pnl
            gross_positions = gross_positions * (1 + r)

            if day in self._backtest_months:
                monthly_return = (net_total_portfolio_value / monthly_start_value) - 1
                monthly_returns.append(monthly_return)
                monthly_portfolio_values.append(net_total_portfolio_value)
                monthly_start_value = net_total_portfolio_value
                pred_returns.append(expected_returns)
                true_returns.append(self._monthly_stock_returns.loc[day].values)
                if day.month_name() == "December":
                    prev_day = day - pd.Timedelta(days=1)
                    X, y = self._dl.load_dataset("hpt", prev_day)
                    grid.fit(X,y)
                    model = grid.best_model_
                    best_params = grid.best_params_
                else:
                    model = MODEL_REGISTRY[model_type](optuna.trial.FixedTrial(best_params), input_features).to(device)
                    if model_type == "mlp":
                        df_prev = self._backtest[self._backtest.index < day].copy()
                        df_prev = df_prev.reset_index()
                        index_col = df_prev.columns[0]
                        df_prev = df_prev.sort_values(by=[index_col, 'Ticker']).set_index(index_col)
                        
                        train_x = torch.from_numpy(df_prev[features].to_numpy()).float().to(device)
                        train_y = torch.from_numpy(df_prev["Target"].to_numpy()).float().to(device)
                        train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=model.batch_size, shuffle=False)
                        model = train_model(model,{"train": train_dataloader},optim.AdamW(model.parameters(),lr=model.lr,weight_decay=model.wd), nn.HuberLoss(delta=0.5),300,0)
                    else:
                        X_train_seq_t, y_train_seq_t = self._prepare_sequential_data(day, features, device, is_inference=False)
                        train_dataloader = DataLoader(TensorDataset(X_train_seq_t, y_train_seq_t), batch_size=model.batch_size, shuffle=False)
                        model = train_model(model,{"train": train_dataloader},optim.AdamW(model.parameters(),lr=model.lr,weight_decay=model.wd), nn.HuberLoss(delta=0.5),500,0)
                model.eval()
                with torch.no_grad():
                    if model_type == "mlp":
                        df_day = self._backtest.loc[[day]].sort_values(by="Ticker")
                        mlp_in = torch.from_numpy(df_day[features].values).float().to(device)
                        pred = model(mlp_in)
                    else:
                        pred_input = self._prepare_sequential_data(day, features, device, is_inference=True)
                        pred = model(pred_input)
                
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                pred_z_scores.append(pred)
                
                df_day_sorted = self._backtest.loc[[day]].sort_values(by="Ticker")
                true_z_scores.append(df_day_sorted["Target"].to_numpy())
                
                preds["Predictions"] = pred
                old_weights = weights
                weights, expected_returns = bl.optimize_portfolio(self._daily_stock_returns[self._daily_stock_returns.index <= day].tail(504), day, preds=preds, y_pred=pred_z_scores, y_true=true_z_scores)
                turnover = self._calculate_turnover(weights, old_weights)
                trans_costs = net_total_portfolio_value * turnover * (TRANSACTION_COSTS + SLIPPAGE)
                net_total_portfolio_value -= trans_costs
                net_positions = weights * net_total_portfolio_value
                gross_positions = weights * gross_total_portfolio_value
                
        total_trans_costs = gross_total_portfolio_value - net_total_portfolio_value
        daily_res = pd.DataFrame({"Portfolio Growth" : daily_portfolio_values, "Returns" : daily_returns}, index=self._backtest_days)

        self._plotter.plot_equity_cruve(title, daily_res, self._snp500_daily_returns)
        plot_type = 2 if model_type == 0 or model_type == "random" else 3
        self._plotter.plot_metrics(title, plot_type, monthly_returns, self._rfr, daily_portfolio_values, net_total_portfolio_value, monthly_portfolio_values, self._backtest_months, self._fama, total_trans_costs, pred_returns, true_returns, pred_z_scores, true_z_scores)
        self._plotter.plot_dd(title, daily_portfolio_values, daily_res)

        self._monte_carlo.monte_carlo_path_sim(title, daily_res, self._rfr, self._snp500_cagr, self._backtest_months)

    def _prepare_sequential_data(self, day, features, device, is_inference=False):
        if is_inference:
            all_months = self._backtest[self._backtest.index <= day].index.unique().sort_values()
            months_12 = all_months[-12:]
            
            df_12 = self._backtest[self._backtest.index.isin(months_12)].copy()
            df_12 = df_12.reset_index()
            index_col = df_12.columns[0]
            df_12 = df_12.sort_values(by=[index_col, 'Ticker']).set_index(index_col)
            
            X_12 = df_12[features].to_numpy()
            X_tensor = torch.from_numpy(X_12).float().to(device)
            
            pred_input = X_tensor.view(12, 30, len(features)).transpose(0, 1)
            return pred_input
        else:
            df_prev = self._backtest[self._backtest.index < day].copy()
            months_prev = df_prev.index.unique().sort_values()
            M = len(months_prev)
            
            df_prev = df_prev.reset_index()
            index_col = df_prev.columns[0]
            df_prev = df_prev.sort_values(by=[index_col, 'Ticker']).set_index(index_col)
            
            X_numpy = df_prev[features].to_numpy()
            y_numpy = df_prev["Target"].to_numpy()
            
            X_full = torch.from_numpy(X_numpy).float().to(device)
            y_full = torch.from_numpy(y_numpy).float().to(device)
            
            X_3d = X_full.view(M, 30, len(features))
            X_trans = X_3d.transpose(0, 1)
            
            X_seq_list = []
            y_seq_list = []
            for t in range(11, M):
                X_seq_list.append(X_trans[:, t-11 : t+1, :])
                y_seq_list.append(y_full[t * 30 : (t+1) * 30])
                
            X_train_seq = torch.cat(X_seq_list, dim=0)
            y_train_seq = torch.cat(y_seq_list, dim=0)
            
            return X_train_seq, y_train_seq