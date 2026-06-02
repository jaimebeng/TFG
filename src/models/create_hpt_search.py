from src.models.create_shallow_models import MODEL_REGISTRY
from src.models.rolling_gridsearch import RollingGridSearch
from src.models.optuna_search import RollingOptunaSearch
import optuna

SEARCH_REGISTRY = {
    "lassoreg": {
        "method": "grid",
        "builder": MODEL_REGISTRY["lassoreg"],
        "wrapper": RollingGridSearch,
        "extra_args": {"min_train_size": 12, "verbose": 0},
    },
    "ridgereg": {
        "method": "grid",
        "builder": MODEL_REGISTRY["ridgereg"],
        "wrapper": RollingGridSearch,
        "extra_args": {"min_train_size": 12, "verbose": 0},
    },
    "elasticnet": {
        "method": "grid",
        "builder": MODEL_REGISTRY["elasticnet"],
        "wrapper": RollingGridSearch,
        "extra_args": {"min_train_size": 12, "verbose": 0},
    },

    "rf": {
        "method": "optuna",
        "builder": MODEL_REGISTRY["rf"],
        "wrapper": RollingOptunaSearch,
        "n_trials": 80,
        "sampler": None,
        "pruner": optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=15
        ),
        "min_train_size": 12,
        "verbose": 0,
    },

    "xgboost": {
        "method": "optuna",
        "builder": MODEL_REGISTRY["xgboost"],
        "wrapper": RollingOptunaSearch,
        "n_trials": 80,
        "sampler": optuna.samplers.TPESampler(seed=42),
        "pruner": optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=15
        ),
        "min_train_size": 12,
        "verbose": 0,
    }

}


def create_search(model_type):
    cfg = SEARCH_REGISTRY[model_type]

    if cfg["method"] == "grid":
        model, param_grid = cfg["builder"]()
        return cfg["wrapper"](model, param_grid, **cfg["extra_args"])

    elif cfg["method"] == "optuna":
        return cfg["wrapper"](cfg["builder"], cfg["min_train_size"], cfg["n_trials"], cfg["sampler"], cfg["pruner"], verbose=cfg["verbose"])
