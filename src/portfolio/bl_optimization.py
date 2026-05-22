from src.data.data_loader import DataLoad
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import cvxpy as cp

class Black_Litterman():

    def __init__(self,tickers, delta = 2.5, gamma = 2.5, tau = 0.025, n_assets = 30):
        self._tickers = tickers
        self._delta = delta
        self._gamma = gamma
        self._tau = tau
        self._n_assets = n_assets

    def _calculate_zscores(self, preds):
        results = pd.DataFrame({"Ticker": self._tickers})
        results["Predictions"] = preds
        results["Ranking"] = results["Predictions"].rank(ascending=True) 
        results["Probabilities"] = (results["Ranking"] - 0.5) / len(results["Ranking"])
        results["Z-Score"] = norm.ppf(results["Probabilities"])

        return results
    
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
    
    def _calculate_pi(self):
        market_weights = np.ones(self._n_assets) / self._n_assets
        pi = self._delta * self.sigma @ market_weights

        return pi

    def _calculate_Q(self, vols, z_scores):
        monthly_vols = vols * np.sqrt(21)
        Q = z_scores * monthly_vols 

        return Q

    def _calculate_omega(self, confidence_factor, sigma):
        conf_mult = (1 - confidence_factor) / confidence_factor
        diag_cov = np.diag(sigma)
        omega = diag_cov * conf_mult

        return omega
    
    def _calculate_expected_returns(self, tau, sigma, pi, omega, Q):
        P = np.eye(30)
        tau = 0.025
        tau_sigma = tau * sigma
        deviation = Q - (P @ pi)
        kalman_gain = tau_sigma @ P.T @ np.linalg.inv(((P @ tau_sigma @ P.T) + omega))
        expected_returns = pi + (kalman_gain @ deviation)

        return expected_returns
    
    def optimize_portfolio(self, df, confidence_factor, verbose = 0):
        df.drop(columns="GSPC", inplace=True)
        results = self._calculate_zscores()
        sigma, vols = self._calculate_sigma(df, verbose)
        pi = self._calculate_pi
        Q = self._calculate_Q(vols, results["Z-Score"])
        omega = self._calculate_omega(confidence_factor, sigma)
        results["Expected Returns"] = self._calculate_expected_returns(self._tau, sigma, pi, omega, Q)

        mu = results["Expected Returns"].to_numpy()
        weights = cp.Variable(self._n_assets)
        risk = cp.quad_form(weights,sigma)
        returns = mu @ weights

        objective = cp.Maximize(returns - ((1/2) * self._gamma * risk))
        constraints = [weights >= -0.1, weights <= 0.1, cp.sum(weights) == 0, cp.norm1(weights) <= 2]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        results["Portfolio Weights"] = weights.value
        if verbose:
            for ticker, value in zip(results["Ticker"], results["Portfolio Weights"]):
                print(f"{ticker}: {value}")

        return results[["Ticker", "Portfolio Weights"]]
