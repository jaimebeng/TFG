# Dataset Card: Frozen Data Snapshot

This document describes the **frozen data snapshot** that accompanies the repository and
makes the empirical results bit-for-bit reproducible. The data is **not committed to Git**
(it is ~400 MB and `data/` is git-ignored); it is published as a separate archive on Zenodo
alongside the code release.

> **Why a frozen snapshot is necessary.** The pipeline ingests prices from Yahoo Finance via
> the `yfinance` library. Yahoo's *adjusted* prices are recomputed on every subsequent split
> and dividend, and constituent metadata changes over time. Re-downloading the data at a later
> date therefore yields **different** historical series and will not reproduce the published
> 2010–2025 results. The only way to reproduce the study exactly is to run the frozen pipeline
> on this frozen snapshot.

## Contents

```
data/
├── raw/         # As-downloaded daily OHLCV per ticker, plus GSPC, market_caps, fama
├── clean/       # Calendar-aligned, gap-filled daily series
├── processed/   # Monthly-resampled, aligned series
└── features/    # Engineered monthly feature panels per ticker
```

| Field | Value |
| :--- | :--- |
| **Universe** | 30 large-cap US equities (GICS-diversified S&P 500 constituents) |
| **Tickers** | AAPL, MSFT, NVDA, GOOG, ORCL, AVGO, AMZN, HD, MCD, NKE, JNJ, UNH, PFE, MRK, CAT, BA, UPS, MMM, XOM, CVX, SLB, JPM, BAC, GS, MS, NEE, LIN, SHW, VZ, CMCSA |
| **Period** | 2010–2025 (16 years); experimentation 2010–2016, out-of-sample backtest 2017–2025 |
| **Frequency** | Daily OHLCV → monthly features / monthly next-month log-return target |
| **Benchmark** | S&P 500 Index (`^GSPC`) |
| **Auxiliary** | Historical market-capitalisation weights; Fama–French 5-factor library + RF (3-month T-bill) |
| **Approx. size** | ~400 MB across all stages |

## Provenance & licensing

- **Price / volume / market-cap data:** retrieved from **Yahoo Finance** via the open-source
  `yfinance` library. This data is redistributed here **solely for academic reproducibility**
  of the accompanying thesis and is subject to
  [Yahoo's Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html).
  It is **not** licensed for commercial use. If you require an unrestricted dataset, regenerate
  it from your own source (set the download flags in `main.py`).
- **Fama–French 5-factor data:** from the
  [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
  which is publicly available.
- The **MIT licence** in this repository covers the **source code only**, not the bundled data.

## How to use it

1. Download the data archive from the Zenodo record (see the DOI in `README.md`).
2. Unzip it into the repository root so that the `data/` directory sits next to `main.py`.
3. In `main.py`, keep `DOWNLOAD = CLEAN = PROCESS = FEATURE = DATASET = False` and `BACKTEST = True`.
4. Run `python main.py`. Outputs are written to `results/`.

## Snapshot date

The snapshot reflects the data as retrieved during the study (mid-2026). It is intentionally
**not** refreshed, for the reproducibility reason explained above.
