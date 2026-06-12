"""
Module for instantiating shallow machine learning regressors and hyperparameter search grids.
Defines constructors and grids/search spaces for classical linear models (Lasso, Ridge,
ElasticNet) using scikit-learn, and tree-based ensemble methods (Random Forest, XGBoost)
using Optuna trial suggestions.
"""

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
    param_grid = {'alpha': np.logspace(-4, -1.5, 15)}
    return Lasso(max_iter=1000, tol=1e-4, selection="random"), param_grid

def ridge_reg():
    param_grid = {'alpha': np.logspace(0, 3, 20)}
    return Ridge(max_iter=1000,tol=1e-4,solver="cholesky"), param_grid

def elastic_net():
    param_grid = {'alpha': np.logspace(-4, -1, 10), 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
    return ElasticNet(max_iter=1000,tol=1e-4), param_grid

def rf_builder(trial):
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 180),
        max_depth=trial.suggest_int("max_depth", 6, 10),
        max_features=trial.suggest_float("max_features", 0.35, 0.55),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 12, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 15, 25),
        n_jobs=-1,
        random_state=42
    )

def xg_builder(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 60, 150),
        max_depth=trial.suggest_int("max_depth", 1, 3),
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 3.0, 12.0, log=True),
        gamma=trial.suggest_float("gamma", 0.02, 0.25),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 0.2, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1.5, 6.0, log=True),
        subsample=trial.suggest_float("subsample", 0.40, 0.70),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.40, 0.75),
        n_jobs=-1,
        random_state=42
    )


MODEL_REGISTRY = {
    "linear": lin_reg,
    "lassoreg": lasso_reg,
    "ridgereg": ridge_reg,
    "elasticnet": elastic_net,
    "rf": rf_builder,
    "xgboost": xg_builder
}
