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
        "n_trials": 150,
        "sampler": None,
        "pruner": None,
        "min_train_size": 12,
        "verbose": 0,
    },

    "xgboost": {
        "method": "optuna",
        "builder": MODEL_REGISTRY["xgboost"],
        "wrapper": RollingOptunaSearch,
        "n_trials": 150,
        "sampler": None,
        "pruner": optuna.pruners.SuccessiveHalvingPruner(min_resource=10, reduction_factor=3),
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
