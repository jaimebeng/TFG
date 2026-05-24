from src.data.data_loader import DataLoad
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import cvxpy as cp

class Black_Litterman():

    def __init__(self, delta = 2.5, gamma = 2.5, tau = 0.025, n_assets = 30):
        self._delta = delta
        self._gamma = gamma
        self._tau = tau
        self._n_assets = n_assets

    def _calculate_zscores(self, preds):
        preds["Ranking"] = preds["Predictions"].rank(ascending=True) 
        preds["Probabilities"] = (preds["Ranking"] - 0.5) / len(preds["Ranking"])
        preds["Z-Score"] = norm.ppf(preds["Probabilities"])
        preds = preds.set_index("Ticker")

        return preds
    
    def _calculate_sigma(self, df, verbose = 0):
        df = df.copy()
        vols = df.std().values
        D = np.diag(vols)

        corr_matrix = df.corr()
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

        N = len(df.columns)
        T = df.shape[0]

        if verbose:
            a = (1 - np.sqrt(N/T))**2
            b = (1 + np.sqrt(N/T))**2
            x = np.linspace(eigenvalues.min(), eigenvalues.max(), 1000)
            density = np.zeros_like(x)
            mask = (x >= a) & (x <= b)

            density[mask] = (np.sqrt((b - x[mask]) * (x[mask] - a)) / (2 * np.pi * (N/T) * x[mask]))
            mp = density

            plt.figure(figsize=(8, 5))
            plt.hist(eigenvalues, bins=50, density=True, alpha=0.5, label="Empirical eigenvalues")
            plt.plot(x, mp, 'r', lw=2, label="Marchenko-Pastur density")
            plt.axvline(a, color='k', linestyle='--', alpha=0.6)
            plt.axvline(b, color='k', linestyle='--', alpha=0.6)
            plt.title("Eigenvalue Spectrum vs Marchenko-Pastur Law")
            plt.xlabel("Eigenvalue")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

        max_eigen = (1 + np.sqrt(N / T)) ** 2
        noise_eigen = eigenvalues <= max_eigen
        cleaned_eigenvalues = eigenvalues.copy()
        cleaned_eigenvalues[noise_eigen] = np.mean(eigenvalues[noise_eigen])
        clean_corr_matrix = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
        diag = np.diag(clean_corr_matrix)
        clean_corr_matrix = clean_corr_matrix / np.sqrt(np.outer(diag,diag))
        clean_cov_matrix = D @ clean_corr_matrix @ D
        sigma = clean_cov_matrix * 21

        return sigma, vols
    
    def _calculate_pi(self, date, sigma):
        dl = DataLoad()
        market_caps = dl.load_dataset("market_caps", date=date)
        total_mcap = np.sum(market_caps.to_numpy())
        market_weights = (market_caps.to_numpy().flatten() / total_mcap).reshape(self._n_assets, 1)

        pi = self._delta * sigma @ market_weights

        return pi

    def _calculate_Q(self, vols, z_scores):
        monthly_vols = vols * np.sqrt(21)
        z = np.asarray(z_scores).reshape(-1,1)
        Q = z * monthly_vols.reshape(-1,1)

        return Q

    def _calculate_omega(self, confidence_factor, sigma, P):
        conf_mult = (1 - confidence_factor) / confidence_factor
        omega = np.diag(np.diag(P @ sigma @ P.T) * conf_mult)

        return omega
    
    def _calculate_expected_returns(self, sigma, pi, omega, Q, P):
        tau_sigma = self._tau * sigma
        deviation = Q - (P @ pi)
        kalman_gain = tau_sigma @ P.T @ np.linalg.inv(((P @ tau_sigma @ P.T) + omega))
        expected_returns = pi + (kalman_gain @ deviation)

        return expected_returns
    
    def optimize_portfolio(self, preds, returns_df, date, confidence_factor, verbose = 0):
        results = self._calculate_zscores(preds)
        sigma, vols = self._calculate_sigma(returns_df, verbose)
        pi = self._calculate_pi(date, sigma)
        results = results.reindex(returns_df.columns, axis=0)
        Q = self._calculate_Q(vols, results["Z-Score"])
        P = np.eye(self._n_assets)
        omega = self._calculate_omega(confidence_factor, sigma, P)
        expected_returns = self._calculate_expected_returns(sigma, pi, omega, Q, P)

        mu = expected_returns
        weights = cp.Variable(self._n_assets)
        risk = cp.quad_form(weights,sigma)
        port_returns = weights.T @ mu

        objective = cp.Maximize(port_returns - ((1/2) * self._gamma * risk))
        constraints = [weights >= -0.1, weights <= 0.1, cp.sum(weights) == 0, cp.norm1(weights) <= 2]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        results["Portfolio Weights"] = weights.value
        if verbose:
            for ticker, value in zip(results.index, results["Portfolio Weights"]):
                print(f"{ticker}: {value}")

        return results[["Portfolio Weights"]]
