import os
import sys
sys.path.append(os.path.abspath(".."))
from src.utils.metrics import bids
import optuna
import numpy as np


class RollingOptunaSearch():

    def __init__(self, model_builder, min_train_size=24, n_trials=30, sampler=None, pruner=None, verbose=1):
        self.model_builder = model_builder
        self.scorer_ = bids
        self.min_train_size = min_train_size
        self.n_trials = n_trials
        self.verbose = verbose
        self.sampler = sampler or optuna.samplers.TPESampler(multivariate=True, seed=42)
        self.pruner = pruner or optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3, min_early_stopping_rate=0)

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
                score = scorer(y_pred, y[train_len:len(Xs[t])])
                scores.append(score)
                current_mean = np.mean(scores)

                trial.report(current_mean, step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            mu = np.mean(scores)
            sigma = np.std(scores) + 1e-6

            # Return the Risk-Adjusted BIDS, this favors high-performing models that don't fluctuate wildly month-to-month
            return mu * (mu / sigma)
        
        return objective

    def fit(self, X, y):

        start_t = max(1, self.min_train_size)
        lens = [(len(X[t-1]), t) for t in range(start_t, len(X))]

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner
        )

        optuna_jobs = -1

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