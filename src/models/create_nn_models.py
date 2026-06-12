"""
Module for constructing Optuna hyperparameter search spaces for PyTorch neural networks.
Defines builder functions that dynamically sample architectural parameters (layer dimensions,
channels, attention heads), learning rates, weight decays, dropout rates, and batch sizes
from predefined ranges using an Optuna trial, returning initialized PyTorch modules.
"""

from src.models.nn_models import MLP, CNNmodel, LSTMmodel, CNN_LSTM, Transformer
from ast import literal_eval

def mlp_builder(trial, input_features):
    hidden = trial.suggest_categorical("hidden_layer_sizes", ["[8]", "[16]", "[32]", "[16, 8]", "[32, 16]"])
    return MLP(
        input_size=input_features,
        hidden_layers = literal_eval(hidden),
        activation = trial.suggest_categorical("activation", ['gelu', 'relu', 'tanh']),
        lr=trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        wd=trial.suggest_float("wd", 1e-3, 1e-1, log=True),
        dropout = trial.suggest_float("dropout", 0.20, 0.50, step=0.05),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]))


def cnn_builder(trial, input_features):
    return CNNmodel(
        input_features=input_features,
        out_channels_l1=trial.suggest_int("out_channels_l1", 16, 32, step=16),
        out_channels_l2=trial.suggest_int("out_channels_l2", 16, 32, step=16),
        dropout=trial.suggest_float("dropout", 0.2, 0.5, step=0.05),
        fc_dim=trial.suggest_categorical("fc_dim", [16, 32]),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        wd=trial.suggest_float("wd", 1e-3, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]))

def lstm_builder(trial, input_features):
    return LSTMmodel(
        input_features=input_features,
        hidden_size=trial.suggest_int("hidden_size", 16, 32, step=8),
        dropout=trial.suggest_float("dropout", 0.3, 0.5, step=0.05),
        fc_dim=trial.suggest_int("fc_dim", 16, 32, step=8),
        lr=trial.suggest_float("lr", 8e-5, 1e-3, log=True),
        wd=trial.suggest_float("wd", 2e-3, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256]))

def cnn_lstm_builder(trial, input_features):
    return CNN_LSTM(
        input_features=input_features,
        cnn_out_channels=trial.suggest_int("cnn_out_channels", 16, 32, step=16), 
        lstm_hidden_dim=trial.suggest_int("lstm_hidden_dim", 16, 64, step=16),
        dropout=trial.suggest_float("dropout", 0.2, 0.5, step=0.05),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        wd=trial.suggest_float("wd", 5e-4, 2e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]))

def transformer_builder(trial, input_features): 
    return Transformer(
        input_features=input_features,
        embed_dim=trial.suggest_int("embed_dim", 16, 32, step=16),
        num_heads=trial.suggest_categorical("num_heads", [1, 2, 4]),
        dropout=trial.suggest_float("dropout", 0.15, 0.45, step=0.05),
        fc_dim=trial.suggest_int("fc_dim", 32, 64, step=32),
        lr=trial.suggest_float("lr", 2e-4, 2e-3, log=True),
        wd=trial.suggest_float("wd", 1e-3, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]))

MODEL_REGISTRY = {
    "mlp": mlp_builder,
    "cnn": cnn_builder,
    "lstm": lstm_builder,
    "cnn_lstm": cnn_lstm_builder,
    "transformer": transformer_builder
}
