from src.models.create_models import *
from src.models.rolling_gridsearch import RollingGridSearch
from src.models.optuna_search import RollingOptunaSearch
import optuna

SEARCH_REGISTRY = {
    "lassoreg": {
        "method": "grid",
        "builder": lasso_reg,
        "wrapper": RollingGridSearch,
        "extra_args": {"window": 12, "verbose": 0},
    },
    "ridgereg": {
        "method": "grid",
        "builder": ridge_reg,
        "wrapper": RollingGridSearch,
        "extra_args": {"window": 12, "verbose": 0},
    },
    "elasticnet": {
        "method": "grid",
        "builder": elastic_net,
        "wrapper": RollingGridSearch,
        "extra_args": {"window": 12, "verbose": 0},
    },

    "rf": {
        "method": "optuna",
        "builder": rf_builder,
        "wrapper": RollingOptunaSearch,
        "n_trials": 220,
        "sampler": None,
        "pruner": optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2,
            reduction_factor=3,
            min_early_stopping_rate=0
        ),
        "min_train_size": 12,
        "verbose": 0,
    },

    "xgboost": {
        "method": "optuna",
        "builder": xg_builder,
        "wrapper": RollingOptunaSearch,
        "n_trials": 260,
        "sampler": optuna.samplers.TPESampler(seed=42),
        "pruner": optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2,
            reduction_factor=3,
            min_early_stopping_rate=0
        ),
        "min_train_size": 12,
        "verbose": 0,
    }
}


def create_search(model_type):
    cfg = SEARCH_REGISTRY[model_type]

    if cfg["method"] == "grid":
        model, param_grid = cfg["builder"]()
        return cfg["wrapper"](model, param_grid,**cfg["extra_args"])

    elif cfg["method"] == "optuna":
        return cfg["wrapper"](cfg["builder"], cfg["min_train_size"], cfg["n_trials"], cfg["sampler"], cfg["pruner"], verbose=cfg["verbose"])
