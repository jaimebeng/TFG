"""
Module for running hyperparameter optimization via Optuna on shallow machine learning models.
Features an expanding-window chronological cross-validation structure with built-in
trial pruning (Successive Halving) to maximize Spearman rank correlation (IC) while reducing
computational overhead.
"""

import os
import sys
sys.path.append(os.path.abspath(".."))
from scipy.stats.mstats import spearmanr
import optuna
import numpy as np


class RollingOptunaSearch():
    """
    Hyperparameter search wrapper for shallow estimators using Optuna.

    This class:
    1. Implements an expanding-window chronological cross-validation scheme.
    2. At each test step, computes the Spearman rank correlation coefficient (IC) between
       predictions and actual next-month returns.
    3. Leverages TPESampler and SuccessiveHalvingPruner to search parameters and prune
       underperforming configurations early based on intermediate average IC values.
    4. Automatically refits the best discovered parameters on the final complete historical
       dataset to prepare for final predictions.
    """

    def __init__(self, model_builder, min_train_size=24, n_trials=30, sampler=None, pruner=None, verbose=1):
        self.model_builder = model_builder
        self.scorer_ = spearmanr
        self.min_train_size = min_train_size
        self.n_trials = n_trials
        self.verbose = verbose
        self.sampler = sampler or optuna.samplers.TPESampler(multivariate=True, seed=42)
        self.pruner = pruner or  optuna.pruners.SuccessiveHalvingPruner(min_resource=7, reduction_factor=3)

    def _build_objective(self, Xs, y, lens):

        scorer = self.scorer_

        def objective(trial):

            model_base = self.model_builder(trial)

            scores = []

            for i, (train_len, t) in enumerate(lens):

                if len(Xs[t]) <= train_len: 
                    continue

                model_base.fit(Xs[t][:train_len], y[:train_len])
                y_pred = model_base.predict(Xs[t][train_len:])
                score = scorer(y_pred, y[train_len:len(Xs[t])])[0]
                scores.append(score)
                current_mean = np.mean(scores)

                trial.report(current_mean, step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            mu = np.mean(scores)

            return mu
        
        return objective

    def fit(self, X, y):

        start_t = max(1, self.min_train_size)
        lens = [(len(X[t-1]), t) for t in range(start_t, len(X))]

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner
        )

        optuna_jobs = 1

        study.optimize(self._build_objective(X, y, lens), n_trials=self.n_trials,n_jobs=optuna_jobs)

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        self.completed_trials = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        self.pruned_trials = sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)

        if self.verbose:
            print(f"Trials: {len(study.trials)} total, {self.completed_trials} complete, {self.pruned_trials} pruned")
            print("Best params:", self.best_params_)
            print("Best score:", self.best_score_)

        self.best_model_ = self.model_builder(optuna.trial.FixedTrial(self.best_params_))
        self.best_model_.fit(X[-1], y)

        return self

    def predict(self, X):
        return self.best_model_.predict(X)