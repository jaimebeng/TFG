# imports
import os
import sys
sys.path.append(os.path.abspath(".."))
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

def lin_reg():
    return LinearRegression(n_jobs=-1)

def lasso_reg():
    param_grid = {'alpha': np.logspace(-4,-1,10)}
    return Lasso(max_iter=1000, tol=1e-4, selection="random"), param_grid

def ridge_reg():
    param_grid = {'alpha': np.logspace(-1, 5, 20)}
    return Ridge(max_iter=1000,tol=1e-4,solver="cholesky"), param_grid

def elastic_net():
    param_grid = {'alpha': np.logspace(-4,3, 20), 'l1_ratio': np.linspace(0.1, 0.9, 9)}
    return ElasticNet(max_iter=1000,tol=1e-4), param_grid

def rf_builder(trial):
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 350),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        max_features=trial.suggest_float("max_features", 0.6, 1.0),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 8),
        min_samples_split=trial.suggest_int("min_samples_split", 4, 8),
        n_jobs=1,
        random_state=42
    )

def xg_builder(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 150, 1200),
        max_depth=trial.suggest_int("max_depth", 5, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 0.8, 1.5, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 0.1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 0.05, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.8, 3.0, log=True),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        n_jobs=1,
        random_state=42
    )

