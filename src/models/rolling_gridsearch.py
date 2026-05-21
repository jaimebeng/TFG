import os
import sys
sys.path.append(os.path.abspath(".."))
from src.utils.metrics import bids
from sklearn.base import clone
from itertools import product
from joblib import Parallel, delayed
import numpy as np

class RollingGridSearch():
    """
    Rolling-window grid search for time-series / expanding window prediction.
    """

    def __init__(self, model, param_grid, min_train_size=24, n_jobs=-1, verbose=1):

        self.model = model
        self.param_grid = param_grid
        self.scorer_ = bids
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
            
            score = self.scorer_(y_pred, y_test)
            scores.append(score)

        mu = np.mean(scores)
        sigma = np.std(scores) + 1e-6
        score = mu * (mu / sigma) # Risk-Adjusted BIDS, will help find more stable parameters, instead of just the peak
        return {"params": params, "score": score}

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
