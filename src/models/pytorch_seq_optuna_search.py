"""
Module for running hyperparameter optimization via Optuna on sequential PyTorch neural networks.
Implements data transformation pipelines that slice 2D flat features into 3D sequential blocks
using a 12-month rolling sequence window, enabling sequential model tuning (CNN, LSTM, Transformer).
"""

import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from src.models.pytorch_optuna_search import train_model

class RollingSeqOptunaSearchPyTorch():
    """
    Hyperparameter search wrapper using Optuna for sequential deep learning models.

    This class handles:
    1. Mapping 2D flat cross-sectional inputs into 3D temporal sequence tensors with a
       12-month historical window (lag-11 to present).
    2. Partitioning sequences chronologically into expanding training windows and corresponding
       out-of-sample monthly validation targets.
    3. Tuning sequential architectures (e.g. LSTM, CNN_LSTM, Transformer) via TPESampler and
       SuccessiveHalvingPruner.
    4. Slicing predictions and validation losses on GPU/CPU to optimize rank-based IC metrics.
    5. Refitting the optimal network configuration on the entire sequential dataset history.
    """

    def _make_sequences(self, X, y):
        M = len(X)
        X_full = X[-1]
        F = X_full.shape[1]
        X_3d = X_full.view(M, 30, F)
        X_trans = X_3d.transpose(0, 1)
        
        X_seq = []
        y_seq = []
        for t in range(M):
            if t < 11:
                X_seq.append(None)
                y_seq.append(None)
            else:
                X_seq.append(X_trans[:, t-11 : t+1, :])
                y_seq.append(y[t * 30 : (t+1) * 30])
                
        return X_seq, y_seq

    def __init__(self, model_builder, input_features, device, min_train_size=24, n_trials=30, sampler=None, pruner=None, verbose=1, hpt_epochs=150):
        self.model_builder = model_builder
        self.input_features = input_features
        self.device = device
        self.min_train_size = min_train_size
        self.n_trials = n_trials
        self.verbose = verbose
        self.sampler = sampler or optuna.samplers.TPESampler(multivariate=True, seed=42)
        self.pruner = pruner or optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=10)

        self.hpt_epochs = hpt_epochs

    def _build_objective(self, X_seq, y_seq, lens):

        def objective(trial):

            scores = []

            for i, (_, t) in enumerate(lens):

                X_train_t = torch.cat(X_seq[11:t], dim=0)
                y_train_t = torch.cat(y_seq[11:t], dim=0)

                X_val_t = X_seq[t]
                y_val_t = y_seq[t]

                model = self.model_builder(trial, self.input_features).to(self.device)

                train_dataloader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=model.batch_size, shuffle=False)
                val_dataloader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=model.batch_size, shuffle=False)

                score = train_model(model, {"train" : train_dataloader, "val" : val_dataloader}, optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd), nn.HuberLoss(delta=0.5), self.hpt_epochs, 0)
                scores.append(score)

                current_mean = np.mean(scores)
                trial.report(current_mean, step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            mu = np.mean(scores)
            return mu
        
        return objective

    def fit(self, X, y):

        X = [torch.from_numpy(x).float().to(self.device, non_blocking=True) for x in X]
        y = torch.from_numpy(y).float().to(self.device, non_blocking=True)

        X_seq, y_seq = self._make_sequences(X, y)

        start_t = max(12, self.min_train_size)
        lens = [((t - 11) * 30, t) for t in range(start_t, len(X_seq))]

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner
        )

        study.optimize(self._build_objective(X_seq, y_seq, lens), n_trials=self.n_trials)

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        self.completed_trials = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        self.pruned_trials = sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)

        if self.verbose:
            print(f"Trials: {len(study.trials)} total, {self.completed_trials} complete, {self.pruned_trials} pruned")
            print("Best params:", self.best_params_)
            print("Best score:", self.best_score_)

        self.best_model_ = self.model_builder(optuna.trial.FixedTrial(self.best_params_), self.input_features).to(self.device)
        X_train_full = torch.cat(X_seq[11:], dim=0)
        y_train_full = torch.cat(y_seq[11:], dim=0)
        train_dataloader = DataLoader(TensorDataset(X_train_full, y_train_full), batch_size=self.best_model_.batch_size, shuffle=False)
        self.best_model_ = train_model(self.best_model_, {"train": train_dataloader}, optim.AdamW(self.best_model_.parameters(), lr=self.best_model_.lr, weight_decay=self.best_model_.wd), nn.HuberLoss(delta=0.5), 500, 1)

        return self

    def predict(self, X):
        self.best_model_.eval()
        with torch.no_grad():
            return self.best_model_(X).cpu().numpy()
