# On the Limits of Low-Frequency OHLCV Signals in Machine Learning-Driven Portfolio Optimization

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Academic Status](https://img.shields.io/badge/status-TFG--UPM-orange.svg)](https://www.upm.es/)
[![DOI](https://img.shields.io/badge/DOI-pending%20Zenodo-lightgrey.svg)](#-citation)

This repository contains the complete source code and reproducible empirical framework for the undergraduate thesis (**Trabajo Fin de Grado**) in Computer Science at **Universidad Politécnica de Madrid (UPM)**:

> **Title:** On the Limits of Low-Frequency OHLCV Signals in Machine Learning-Driven Portfolio Optimization  
> **Author:** Jaime Bengoechea Pardo  
> **Tutor:** Antonio García Dopico (DATSI)  
> **Date:** June 2026  

> [!NOTE]
> The code in `src/` and `main.py`, the notebooks, and the thesis PDF are the artifacts as submitted for examination, preserved unchanged. This repository wraps them with the materials needed to make the work publicly citable and reproducible (pinned dependencies, a frozen data snapshot, citation metadata). It does not modify the submitted research.

---

## 🎯 Key Finding (read this first)

This is a documented negative result, and that honesty is the contribution. On a universe of 30 large-cap US equities (2010 to 2025), machine-learning models trained on monthly features derived from daily OHLCV data **do not** produce economically meaningful, net-of-cost improvements over a no-machine-learning Black–Litterman baseline. The best model (an MLP) reaches Sharpe 0.96, but a passive **Prior-Only** Black–Litterman portfolio, which uses no ML views but the same Marchenko–Pastur covariance cleaning, reaches Sharpe 0.94 at roughly a third of the transaction cost. More strikingly, a naive **equally weighted 1/30 portfolio** reaches Sharpe 0.93, essentially tying the entire optimized apparatus. The driver of risk-adjusted performance is risk management (covariance cleaning plus Bayesian shrinkage), not the predictive signal; the optimized strategies' real edge over 1/30 is in drawdown control (around -25.8% vs -34%), not return. Any absolute "beats the S&P 500" reading must be tempered by the survivorship caveat in [Limitations and Scope](#-limitations--scope).

---

## 📌 Abstract

Over the past decades, the application of Machine Learning (ML) models in quantitative finance has grown exponentially. However, financial return series are characterized by an extremely low signal-to-noise ratio and non-stationarity, exposing predictive algorithms to a severe risk of backtest overfitting (the data-snooping problem).

This study conducts a rigorous, empirical stress-test on the limits of predictability based on low-frequency technical features (monthly features calculated from daily OHLCV price-volume data) for optimized portfolio construction. We evaluate whether ML-driven predictions can generate economically meaningful portfolio improvements when integrated with advanced asset allocation frameworks, specifically **Markowitz Mean-Variance Optimization (MVO)** and the **Black–Litterman** model (using ML predictions as subjective "views").

Our empirical framework retrains and tests models over a 16-year horizon (2010 to 2025) on a universe of 30 large-cap US equities. To combat hyperparameter overfitting, a strict pre-backtest model promotion gate (Spearman Rank Information Coefficient, $\text{Rank IC} \ge 0.05$) is enforced on a 2016 hold-out validation window. Promoted forecasts are passed into a portfolio optimizer under realistic institutional constraints (dollar neutrality, leverage caps, and transaction costs with slippage models).

**Key result:** Out-of-sample simulation (2017 to 2025) reveals that while a non-linear Multilayer Perceptron (MLP) achieves strong risk-adjusted performance (Sharpe 0.96 vs S&P 500 0.72), a passive Black-Litterman Prior-Only benchmark (which uses no machine learning views but benefits from spectral covariance cleaning) achieves an equivalent Sharpe of 0.94. The active MLP portfolio suffers from high turnover, incurring cumulative transaction costs of **15.13%** compared to only **4.67%** for the Prior-Only portfolio. The majority of active alpha is eroded by trading friction, and the primary driver of real-world portfolio utility is robust covariance estimation and risk management rather than the predictive edge of daily price-volume ML models.

---

## 📐 Mathematical Formulation

The following core frameworks are implemented inside the modular pipeline.

### 1. The Black–Litterman Asset Allocation Model
The Black–Litterman model blends the market equilibrium prior with subjective investor views using a Bayesian updating formulation. The posterior expected return vector $\hat{\Pi}$ is computed as:

$$\hat{\Pi} = \left[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q\right]$$

Here $\Pi$ is the vector of prior equilibrium returns (from reverse optimization on market-cap weights), $\Sigma$ the regularized covariance matrix, $\tau$ a scaling parameter, $P$ the link matrix, $Q$ the vector of ML-predicted views, and $\Omega$ the diagonal view-uncertainty matrix (dynamically calibrated from rolling Rank IC).

### 2. Spearman Rank Information Coefficient (Rank IC)
The Information Coefficient at month $t$ is the Spearman rank correlation between predicted returns $\hat{R}_{i,t+1}$ and realized returns $R_{i,t+1}$ across the $N$ stocks, computed as the Pearson correlation of their (tie-corrected) ranks:

$$\text{Rank IC}_t = \frac{\operatorname{cov}\!\big(\operatorname{rank}(\hat{R}_{\cdot,t+1}),\ \operatorname{rank}(R_{\cdot,t+1})\big)}{\sigma_{\operatorname{rank}(\hat{R})}\,\sigma_{\operatorname{rank}(R)}}$$

and the reported value is the time average $\overline{\text{IC}} = \frac{1}{T}\sum_t \text{Rank IC}_t$. In code this is `scipy.stats.spearmanr` for the shallow models and an equivalent differentiable Pearson-on-ranks implementation for the neural networks. A strict filter of $\overline{\text{IC}} \ge 0.05$ on the 2016 validation window is required for model promotion.

### 3. Fama–French 5-Factor Risk Attribution
To isolate risk-adjusted alpha from exposure to standard risk premia, out-of-sample portfolio excess returns are regressed against the **Fama–French 5-factor** model:

$$R_{p,t} - R_{f,t} = \alpha + \beta_1 (Mkt - R_f)_t + \beta_2 SMB_t + \beta_3 HML_t + \beta_4 RMW_t + \beta_5 CMA_t + \epsilon_t$$

where $SMB$ (size), $HML$ (value), $RMW$ (profitability, robust-minus-weak), and $CMA$ (investment, conservative-minus-aggressive) are the standard factor premia, and $\alpha$ is the risk-adjusted active return.

### 4. Newey–West HAC Adjustments
Because monthly rebalanced returns over overlapping horizons exhibit heteroskedasticity and autocorrelation, ordinary standard errors are downward biased. We use the **Newey-West HAC** covariance estimator for robust $t$-statistics:

$$\hat{V}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^L \left(1 - \frac{j}{L+1}\right) \left(\hat{\Gamma}_j + \hat{\Gamma}_j^T\right)$$

where $\hat{\Gamma}_j$ is the sample autocovariance of residuals at lag $j$ and $L$ the bandwidth.

### 5. Marchenko–Pastur Covariance Cleaning
Empirical covariance matrices from finite samples are dominated by noise. Eigenvalues falling inside the Marchenko–Pastur noise band $[\lambda_-, \lambda_+]$, with $\lambda_\pm = \sigma^2 (1 \pm \sqrt{N/T})^2$, are treated as noise and replaced by their average, preserving total variance while removing spurious correlation structure.

---

## 📊 Summary of Out-of-Sample Results (2017 to 2025)

Under a transaction-fee model of 10 bps (5 bps commission plus 5 bps slippage):

| Portfolio / Benchmark | Cumulative Return | CAGR | Sharpe | Sortino | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S&P 500 Index** | 200.39% | 13.00% | 0.72 | 1.10 | -33.92% |
| **Equally Weighted (1/30)** | **323.41%** | **17.39%** | 0.93 | 1.50 | -34.46% |
| **Mean-Variance (MVO)** | -22.47% | -2.78% | -0.29 | -0.38 | -40.59% |
| **BL Prior-Only (No ML)** | 236.94% | 14.45% | 0.94 | 1.66 | **-25.80%** |
| **BL MLP (Active ML Views)**| 244.89% | 14.75% | **0.96** | **1.70** | -25.82% |
| **BL ElasticNet (Active)** | 129.89% | 9.70% | 0.58 | 0.97 | -25.16% |
| **BL XGBoost (Active)** | 126.31% | 9.51% | 0.55 | 0.93 | -27.06% |
| **BL Random Forest (Active)** | 109.21% | 8.55% | 0.50 | 0.81 | -27.46% |

(ElasticNet, XGBoost, and Random Forest are the representative linear and tree models; the other linear models fall in the same 0.50 to 0.58 Sharpe cluster.)

**Transaction costs are decisive.** The Prior-Only baseline trades slowly and accumulates only **4.67%** in cumulative cost, whereas the active MLP accumulates **15.13%** and XGBoost is the most expensive strategy in the study at **17.46%**. The active models' small gross edge is therefore almost entirely eroded by trading friction.

**Statistical reading.** HAC-adjusted $t$-statistics of the net return drifts are significant for Prior-Only (3.26), MLP (3.19), and Equally-Weighted (4.18), but fall to 1.8 to 2.1 for individual linear and tree models. Fama–French 5-factor alphas of the linear and tree strategies (2 to 3% annualised) are statistically **insignificant** ($p$ around 0.5 to 0.7). The deep sequential models (CNN 0.038, LSTM 0.029, CNN-LSTM 0.040, Transformer 0.029) failed the 0.05 Rank-IC promotion gate; only the MLP cleared it (0.061).

> **Note (2026):** these figures match the submitted thesis. An earlier version of this README contained a results table from a different run (it understated the 1/30 portfolio and overstated the linear/tree models); it has been corrected here.

---

## ⚠️ Limitations and Scope

This is an honest benchmark; its boundaries are stated explicitly so the results are not over-read.

- **Survivorship and look-ahead in the universe.** The 30 tickers are a fixed list of companies that are large-cap today (for example NVDA, AVGO). This introduces survivorship bias. The bias inflates all strategies roughly equally, so the relative conclusion of this study, that ML views add little over the Prior-Only baseline net of costs, is robust. However, any absolute "beats the S&P 500" claim must be read as caveated, not as a tradeable result. This is a fixed-universe stress-test of technical signals, not a survivorship-free alpha-discovery study.
- **Narrow universe and low frequency.** 30 US large-cap equities at monthly granularity is the most informationally efficient tier of the market. Findings should not be generalised to small-caps, other markets, or higher frequencies.
- **Deep-model results are partly capacity-bound and implementation-bound.** The sequential networks (CNN, LSTM, Transformer) were trained on a data-scarce panel (about 2,160 observations) and, by the author's own analysis, lacked peak-validation early-stopping checkpointing. Their failure to clear the gate should be read as "not supported in this regime and setup," not as evidence that such architectures cannot work in general.
- **Single validation window.** The promotion gate is evaluated on a single 2016 hold-out year.

---

## 🔁 Data Availability and Reproducibility

The pipeline is deterministic (global seed `42`), but the input data is not, if re-downloaded: Yahoo Finance adjusted prices are recomputed on every later split or dividend, so fetching the data again will not reproduce the 2010 to 2025 numbers. Exact reproduction therefore requires the frozen data snapshot.

- **Frozen data snapshot** (about 400 MB) is published as a separate archive on the Zenodo record (it is git-ignored here). See [`DATASET.md`](DATASET.md) for contents, provenance, and licensing.
- **To reproduce the published results:**
  1. Download the data archive from Zenodo (DOI below) and unzip it into the repo root so `data/` sits next to `main.py`.
  2. In `main.py`, keep `DOWNLOAD = CLEAN = PROCESS = FEATURE = DATASET = False` and `BACKTEST = True`.
  3. Run `python main.py`; results are written to `results/`.
- **To regenerate the data from scratch** (this will differ from the published run, for the reason above): set the corresponding flags in `main.py` to `True`.

> The MIT licence covers the code only. The bundled price data originates from Yahoo Finance and is provided for academic reproducibility under their terms; Fama–French factors are from the public Kenneth R. French Data Library.

---

## 📂 Repository Structure

```directory
.
├── src/                        # Modular source package (as submitted)
│   ├── data/                   # Download, clean, process, feature engineering
│   ├── models/                 # Model registry, Optuna rolling search, PyTorch nets
│   ├── portfolio/              # Black-Litterman + Marchenko-Pastur optimization
│   ├── backtest/               # Walk-forward backtest loop + Monte-Carlo simulator
│   └── utils/                  # Metrics, transformers, plotting
├── notebooks/                  # Prototyping notebooks (01_exploration to 07_monte_carlo)
├── results/                    # Per-strategy plots and metrics (committed)
├── paper/                      # Condensed preprint (LaTeX) derived from the thesis
├── main.py                     # Central execution pipeline (run flags + ticker universe)
├── environment.yml             # Conda environment for exact reproduction (Python 3.11)
├── requirements.txt            # Pinned pip dependencies (mirror of environment.yml)
├── CITATION.cff                # Machine-readable citation metadata
├── DATASET.md                  # Data card for the frozen snapshot (Zenodo)
├── LICENSE                     # MIT (code only)
└── README.md
```

> Not in the Git repository: the raw and processed `data/` (about 400 MB, on Zenodo), and the thesis and defense PDFs (large binaries, on Zenodo and the UPM repository). See [Related outputs](#-related-outputs).

---

## ⚙️ Installation and Setup

**Prerequisites:** Python **3.11** (the study was run on CPython 3.11.15) and conda (recommended).

### Option A: conda (exact reproduction)
```bash
git clone https://github.com/jaimebeng/TFG.git
cd TFG
conda env create -f environment.yml
conda activate TFG
```

### Option B: pip
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the pipeline

`main.py` is controlled by boolean flags at the top of the file:

```python
DOWNLOAD = False  # Download historical price and macro data (will differ from frozen snapshot)
CLEAN    = False  # Data cleaning routines
PROCESS  = False  # Process and align datasets
FEATURE  = False  # Compute technical indicators
DATASET  = False  # Format datasets for model ingestion
BACKTEST = True   # Run out-of-sample backtests and generate results
```

```bash
python main.py
```

---

## 🎓 Citation

If you use this codebase or refer to the findings, please cite the thesis:

```bibtex
@thesis{bengoechea2026limits,
  author       = {Jaime Bengoechea Pardo},
  title        = {On the Limits of Low-Frequency OHLCV Signals in Machine Learning-Driven Portfolio Optimization},
  type         = {Trabajo Fin de Grado},
  institution  = {Universidad Politécnica de Madrid},
  school       = {Escuela Técnica Superior de Ingenieros Informáticos},
  year         = {2026},
  month        = {June},
  note         = {Supervised by Antonio García Dopico. Archived at Zenodo, DOI: <to be inserted on release>}
}
```

> **DOI:** A permanent Zenodo DOI will be minted on the first tagged release and inserted here, in the BibTeX `note`, in `CITATION.cff`, and in the DOI badge above.

---

## 🔗 Related outputs

- **Thesis (full, 156 pp):** archived on Zenodo and deposited in the UPM institutional repository (links to be added on release).
- **Preprint:** a condensed paper derived from the thesis lives in [`paper/`](paper/) (arXiv link to be added on submission).

---

## 📄 License

The source code in this repository is licensed under the MIT License, see [LICENSE](LICENSE). The bundled data is not covered by this licence (see [Data Availability](#-data-availability--reproducibility) and [`DATASET.md`](DATASET.md)).
