# =============================================================================
# stocktest.py
# PURPOSE: Linear Regression numerical baseline.
#   - Downloads AAPL daily prices from Yahoo Finance (2018–2024)
#   - Saves prices to data/AAPL_prices.csv for use by all other models
#   - Predicts next-day return using lagged returns only (no sentiment)
#   - Reports Directional Accuracy, RMSE, and MAE
#
# This is an intentionally simple baseline. Its purpose is NOT high accuracy
# but to establish a numerical-only benchmark against which sentiment-
# augmented models are compared.
# =============================================================================

import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ───────────────────────────────────────────────────────────────────
TICKER      = "AAPL"
START_DATE  = "2018-01-01"
END_DATE    = "2024-01-01"
TEST_RATIO  = 0.2          # last 20% of data used for testing
N_LAGS      = 5            # number of lagged return features

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
PRICES_PATH = os.path.join(DATA_DIR, "AAPL_prices.csv")


# ── Step 1: Download & save price data ───────────────────────────────────────
def download_prices() -> pd.DataFrame:
    """
    Download AAPL adjusted closing prices from Yahoo Finance.
    Saves to data/AAPL_prices.csv so all other models use identical data.
    """
    print(f"Downloading {TICKER} prices from Yahoo Finance ({START_DATE} → {END_DATE})...")
    os.makedirs(DATA_DIR, exist_ok=True)

    raw = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present (yfinance >= 0.2)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    prices = raw[["Close"]].copy()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    prices.to_csv(PRICES_PATH)
    print(f"  Saved {len(prices):,} rows to {PRICES_PATH}\n")
    return prices


# ── Step 2: Feature engineering ──────────────────────────────────────────────
def prepare_features(prices: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """
    Compute daily returns and create lagged return features.
    Target variable: next-day return (shifted -1).
    """
    df = prices.copy()
    df["return"] = df["Close"].pct_change()

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)

    df["target"] = df["return"].shift(-1)
    df = df.dropna()
    return df


# ── Step 3: Train / test split ────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    feature_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    X = df[feature_cols].values
    y = df["target"].values

    split = int(len(X) * (1 - TEST_RATIO))
    return X[:split], X[split:], y[:split], y[split:]


# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of days where predicted direction matches actual direction."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    da   = directional_accuracy(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    print(f"\n{'=' * 45}")
    print(f"  Results: {model_name}")
    print(f"{'=' * 45}")
    print(f"  Directional Accuracy : {da:.4f}")
    print(f"  RMSE                 : {rmse:.6f}")
    print(f"  MAE                  : {mae:.6f}")
    print(f"{'=' * 45}\n")

    return {"model": model_name, "DA": da, "RMSE": rmse, "MAE": mae}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  stocktest.py  —  Linear Regression Baseline")
    print("=" * 50 + "\n")

    # 1. Download prices (also saves AAPL_prices.csv for other models)
    prices = download_prices()

    # 2. Feature engineering
    df = prepare_features(prices)
    print(f"Dataset: {len(df):,} rows after feature engineering.")

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples\n")

    # 4. Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Predict & evaluate
    y_pred = model.predict(X_test)
    results = evaluate(y_test, y_pred, "Linear Regression (Numerical Only)")

    print("Interpretation:")
    print("  Directional Accuracy > 0.50 = better than random guessing.")
    print("  RMSE / MAE are in return units (e.g. 0.015 = 1.5% error per day).")
    print("  This flat-prediction baseline motivates introducing sentiment.\n")

    return results


if __name__ == "__main__":
    main()