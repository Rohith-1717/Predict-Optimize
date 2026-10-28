"""
src/data.py
Data acquisition, validation, preprocessing, and feature engineering
for Indian Equities portfolio optimization.
"""

import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import yfinance as yf

# 20 Leading Indian Equities from the NIFTY 50 representing diversified sectors
DEFAULT_INDIAN_TICKERS = [
    "RELIANCE.NS",    # Energy / Conglomerate
    "TCS.NS",         # IT Services
    "HDFCBANK.NS",    # Private Banking
    "INFY.NS",        # IT Services
    "ICICIBANK.NS",   # Private Banking
    "BHARTIARTL.NS",  # Telecom
    "ITC.NS",         # Consumer Goods / Tobacco
    "LT.NS",          # Infrastructure / Engineering
    "KOTAKBANK.NS",   # Private Banking
    "HINDUNILVR.NS",  # FMCG
    "AXISBANK.NS",    # Private Banking
    "SBIN.NS",        # Public Sector Banking
    "BAJFINANCE.NS",  # Financial Services / NBFC
    "ASIANPAINT.NS",  # Consumer Paints
    "HCLTECH.NS",     # IT Services
    "TITAN.NS",       # Consumer Discretionary
    "SUNPHARMA.NS",   # Pharmaceuticals
    "WIPRO.NS",       # IT Services
    "MARUTI.NS",      # Automobile
    "NTPC.NS",        # Power Generation
]


def download_indian_equities(
    tickers: Optional[List[str]] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    output_path: Optional[str] = "data/raw/indian_equities_2015_2025.csv",
) -> Tuple[pd.DataFrame, bool]:
    """
    Download daily adjusted closing prices for Indian equities via Yahoo Finance.
    Saves to CSV with columns: [date, ticker, adjusted_close].
    Returns (df, is_synthetic).
    """
    if tickers is None:
        tickers = DEFAULT_INDIAN_TICKERS

    print(f"Attempting to download data for {len(tickers)} Indian equities from {start_date} to {end_date}...")
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if raw.empty or "Close" not in raw:
            raise ValueError("Downloaded dataset is empty or missing 'Close' column.")
        
        close_df = raw["Close"]
        # If single ticker, close_df is Series; format to DataFrame
        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame(name=tickers[0])

        # Fill minor missing values forward, then backward
        close_df = close_df.ffill().bfill()

        # Check that we have valid data for most tickers
        valid_cols = [col for col in close_df.columns if close_df[col].count() > 200]
        if len(valid_cols) < 5:
            raise ValueError(f"Insufficient valid tickers downloaded: only {len(valid_cols)} found.")

        close_df = close_df[valid_cols]

        # Reshape to long format: date, ticker, adjusted_close
        long_df = close_df.reset_index().melt(
            id_vars=["Date"], var_name="ticker", value_name="adjusted_close"
        )
        long_df.rename(columns={"Date": "date"}, inplace=True)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")
        long_df = long_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            long_df.to_csv(output_path, index=False)
            print(f"Saved real market data ({len(long_df)} rows, {len(valid_cols)} tickers) to {output_path}")

        return long_df, False

    except Exception as e:
        print(f"Warning: Real data download failed ({e}). Generating synthetic Indian equities test data.")
        synthetic_df = generate_synthetic_data(tickers, start_date, end_date)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            synthetic_df.to_csv(output_path, index=False)
            print(f"Saved synthetic test data to {output_path}")
        return synthetic_df, True


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Import a CSV containing columns 'date,ticker,adjusted_close'.
    Validates formatting and sorts chronologically.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"date", "ticker", "adjusted_close"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {required_cols}, found {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "adjusted_close"])
    df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    return df


def generate_synthetic_data(
    tickers: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic equity price paths with correlated geometric Brownian motion
    and GARCH-like volatility clustering. Explicitly marked as synthetic software test data.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    T = len(dates)
    N = len(tickers)

    # Random correlation matrix via Wishart-like structure
    A = np.random.randn(N, N)
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    # Realistic Indian equity parameters: ~12-16% annual drift, 20-30% annual volatility
    dt = 1.0 / 252.0
    mu = np.random.uniform(0.10, 0.18, size=N) * dt
    sigma = np.random.uniform(0.18, 0.32, size=N) * np.sqrt(dt)

    L = np.linalg.cholesky(corr)
    # Generate log-returns with t-distribution fat tails (df=6)
    t_shocks = np.random.standard_t(df=6, size=(T - 1, N)) / np.sqrt(6 / 4)
    corr_shocks = t_shocks @ L.T

    log_returns = mu + sigma * corr_shocks
    initial_prices = np.random.uniform(200, 3000, size=N)

    prices = np.zeros((T, N))
    prices[0] = initial_prices
    for t in range(1, T):
        prices[t] = prices[t - 1] * np.exp(log_returns[t - 1])

    records = []
    for i, ticker in enumerate(tickers):
        for t, date in enumerate(dates):
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "adjusted_close": float(prices[t, i]),
            })

    df = pd.DataFrame(records)
    return df.sort_values(by=["date", "ticker"]).reset_index(drop=True)


def compute_technical_features(price_pivot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute strictly backward-looking technical indicators to prevent look-ahead bias:
    - ret_5d: 5-day return
    - ret_10d: 10-day return
    - ret_20d: 20-day return
    - ret_60d: 60-day return
    - vol_20d: 20-day rolling annualized volatility
    - rsi_14d: 14-day RSI (scaled 0-1)

    Target:
    - target_ret: Next-day return r_{t+1} = (P_{t+1} / P_t) - 1.

    Returns:
    - feature_dict: Dict mapping feature_name -> DataFrame (dates x tickers)
    - target_returns: DataFrame (dates x tickers) of next-day returns
    """
    # Daily simple returns
    daily_rets = price_pivot.pct_change()

    # Next-day return (TARGET): Shift backward so that at date t, target is return realized from t to t+1
    target_returns = daily_rets.shift(-1)

    features = {}
    features["ret_5d"] = price_pivot.pct_change(5)
    features["ret_10d"] = price_pivot.pct_change(10)
    features["ret_20d"] = price_pivot.pct_change(20)
    features["ret_60d"] = price_pivot.pct_change(60)
    features["vol_20d"] = daily_rets.rolling(20).std() * np.sqrt(252)

    # 14-day RSI calculation
    delta = price_pivot.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    features["rsi_14d"] = rsi / 100.0  # normalize to [0, 1]

    # Find valid date indices where all features and target are non-NaN
    valid_mask = ~target_returns.isna().any(axis=1)
    for feat_name, feat_df in features.items():
        valid_mask = valid_mask & (~feat_df.isna().any(axis=1))

    # Slice valid dates
    target_returns = target_returns.loc[valid_mask]
    for feat_name in features:
        features[feat_name] = features[feat_name].loc[valid_mask]

    return features, target_returns


def build_tensor_dataset(
    features: Dict[str, pd.DataFrame],
    target_returns: pd.DataFrame,
    rolling_cov_window: int = 60,
    shrinkage: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Construct aligned feature arrays, target return arrays, and rolling covariance matrices.
    
    Returns:
    - X: Shape (T, N, F) where T is time steps, N is tickers, F is number of features
    - y: Shape (T, N) of realized next-day returns
    - covs: Shape (T, N, N) of shrunk rolling covariance matrices
    - dates: List of date strings
    - tickers: List of ticker strings
    """
    dates = target_returns.index.tolist()
    tickers = target_returns.columns.tolist()
    T = len(dates)
    N = len(tickers)
    feat_keys = list(features.keys())
    F = len(feat_keys)

    X = np.zeros((T, N, F), dtype=np.float32)
    for f_idx, key in enumerate(feat_keys):
        X[:, :, f_idx] = features[key].values.astype(np.float32)

    y = target_returns.values.astype(np.float32)

    # Compute rolling covariance matrices using past daily returns up to day t (strictly backward-looking)
    covs = np.zeros((T, N, N), dtype=np.float32)
    
    # We estimate covariance using the past `rolling_cov_window` daily returns
    # Compute daily return history from target returns shifted back
    daily_rets = target_returns.copy()
    
    for t_idx in range(T):
        if t_idx < rolling_cov_window:
            # For initial window, use available history or simple diagonal
            sub_rets = y[max(0, t_idx - rolling_cov_window) : t_idx + 1]
            if len(sub_rets) > 5:
                sample_cov = np.cov(sub_rets, rowvar=False)
            else:
                sample_cov = np.eye(N) * 0.0004  # ~25% annual vol default
        else:
            sub_rets = y[t_idx - rolling_cov_window : t_idx]
            sample_cov = np.cov(sub_rets, rowvar=False)

        # Apply shrinkage towards diagonal identity to guarantee well-conditioned positive definiteness
        diag_mean = np.trace(sample_cov) / N
        shrunk_cov = (1.0 - shrinkage) * sample_cov + shrinkage * np.eye(N) * diag_mean
        # Add small ridge epsilon for strict positive definiteness
        shrunk_cov += np.eye(N) * 1e-5
        covs[t_idx] = shrunk_cov.astype(np.float32)

    return X, y, covs, [str(d) for d in dates], tickers
