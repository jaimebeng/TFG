"""
Module for running parameter grid searches using cross-validation over time series.
Evaluates models using an expanding-window scheme where the training set grows at each
temporal step, preventing lookahead bias while assessing out-of-sample prediction performance.
"""

import os
import sys
sys.path.append(os.path.abspath(".."))
from scipy.stats.mstats import spearmanr
from sklearn.base import clone
from itertools import product
from joblib import Parallel, delayed
import numpy as np

class RollingGridSearch():
    """
    Grid search cross-validation wrapper using an expanding-window time-series scheme.

    Instead of randomized k-fold splits, this class evaluates candidate parameters
    chronologically:
    1. At each timestep `t`, the training set is composed of all historical observations
       accumulated up to `t-1` (which expands over time).
    2. The candidate model is trained on this expanding dataset and tested on the new,
       unseen observations of timestep `t`.
    3. Hyperparameter combinations are evaluated in parallel, using the mean Spearman rank
       correlation coefficient (information coefficient) across all validation windows as
       the scoring metric.
    4. The best performing parameter configuration is selected and refitted on the full
       historical dataset.
    """

    def __init__(self, model, param_grid, min_train_size=24, n_jobs=-1, verbose=1):

        self.model = model
        self.param_grid = param_grid
        self.scorer_ = spearmanr
        self.min_train_size = min_train_size
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _param_combinations(self):
        keys, values = list(self.param_grid.keys()), list(self.param_grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def _evaluate_params(self, params, X, y):
        scores = []
        model_step = clone(self.model)
        model_step.set_params(**params)

        for t in range(max(1, self.min_train_size), len(X)):

            X_prev = X[t-1]
            X_curr = X[t]

            n_prev = len(X_prev)

            X_train = X_curr[:n_prev]
            y_train = y[:n_prev]

            X_test = X_curr[n_prev:]
            y_test = y[n_prev:len(X_curr)]

            if len(X_test) == 0:
                continue

            model_step.fit(X_train, y_train)
            y_pred = model_step.predict(X_test)
            
            score = self.scorer_(y_pred, y_test)[0]
            scores.append(score)

        mu = np.mean(scores)
        return {"params": params, "score": mu}

    def fit(self, X, y):
        param_list = list(self._param_combinations())

        if self.verbose:
            print("Running grid search...")

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._evaluate_params)(params, X, y)
            for params in param_list
        )

        self.results_ = results
        best = max(results, key=lambda x: x["score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["score"]

        if self.verbose:
            print("Best params:", self.best_params_)
            print("Best score:", self.best_score_)

        X_full = X[-1]

        self.best_model_ = clone(self.model)
        self.best_model_.set_params(**self.best_params_)
        self.best_model_.fit(X_full, y)

        return self

    def predict(self, X):
        return self.best_model_.predict(X)   
