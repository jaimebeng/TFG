"""
Module providing a registry and factory for PyTorch deep learning hyperparameter searches.
Maps neural network model types to their appropriate search wrapper (RollingOptunaSearchPyTorch
for flat MLP inputs, RollingSeqOptunaSearchPyTorch for sequential 3D inputs) and parameters.
"""

import optuna
from src.models.create_nn_models import MODEL_REGISTRY
from src.models.pytorch_optuna_search import RollingOptunaSearchPyTorch
from src.models.pytorch_seq_optuna_search import RollingSeqOptunaSearchPyTorch

SEARCH_REGISTRY = {
    "mlp": {
        "builder": MODEL_REGISTRY["mlp"],
        "wrapper": RollingOptunaSearchPyTorch,
        "sampler": None,
        "pruner": None,
        "min_train_size": 12,
        "n_trials": 100,
        "verbose": 0
    },
    "cnn": {
        "builder": MODEL_REGISTRY["cnn"],
        "wrapper": RollingSeqOptunaSearchPyTorch,
        "sampler": None,
        "pruner": None,
        "min_train_size": 12,
        "n_trials": 100,
        "verbose": 0
    },
    "lstm": {
        "builder": MODEL_REGISTRY["lstm"],
        "wrapper": RollingSeqOptunaSearchPyTorch,
        "sampler": None,
        "pruner": None,
        "min_train_size": 12,
        "n_trials": 100,
        "verbose": 0
    },
    "cnn_lstm": {
        "builder": MODEL_REGISTRY["cnn_lstm"],
        "wrapper": RollingSeqOptunaSearchPyTorch,
        "sampler": None,
        "pruner": None,
        "min_train_size": 12,
        "n_trials": 100,
        "verbose": 0
    },
    "transformer": {
        "builder": MODEL_REGISTRY["transformer"],
        "wrapper": RollingSeqOptunaSearchPyTorch,
        "sampler": None,
        "pruner": None,
        "min_train_size": 12,
        "n_trials": 100,
        "verbose": 0
    }
}


def create_pytorch_search(model_type, device = None, input_features = None):
    cfg = SEARCH_REGISTRY[model_type]

    return cfg["wrapper"](
        cfg["builder"],
        input_features,
        device,
        cfg["min_train_size"],
        cfg["n_trials"],
        cfg["sampler"],
        cfg["pruner"],
        verbose=cfg["verbose"]
    ) 
