# =============================================================================
# arima.py
# PURPOSE: ARIMA time-series numerical baseline.
#   - Uses AAPL daily returns (no sentiment, no external features)
#   - Walks forward one day at a time (expanding window)
#   - Reports Directional Accuracy, RMSE, and MAE
#
# ARIMA answers the question:
#   "Based solely on how this stock has behaved historically,
#    what is the most likely next-day return?"
#
# This is the gold-standard statistical baseline in financial forecasting.
# Combined with the Linear Regression baseline (stocktest.py), it ensures
# that any sentiment improvement holds across both ML and statistical paradigms.
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")   # suppress ARIMA convergence warnings

# ── Config ───────────────────────────────────────────────────────────────────
ARIMA_ORDER = (1, 0, 1)    # (p, d, q) — standard for stationary return series
TEST_RATIO  = 0.2

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
PRICES_PATH = os.path.join(DATA_DIR, "AAPL_prices.csv")


# ── Step 1: Load prices & compute returns ────────────────────────────────────
def load_returns() -> pd.Series:
    """Load saved AAPL prices and compute daily returns."""
    prices  = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    # Flatten MultiIndex columns if present
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] for col in prices.columns]

    returns = prices["Close"].pct_change().dropna()
    print(f"Loaded {len(returns):,} daily returns  "
          f"({returns.index[0].date()} → {returns.index[-1].date()})\n")
    return returns


# ── Step 2: Walk-forward ARIMA forecasting ───────────────────────────────────
def walk_forward_arima(returns: pd.Series) -> tuple:
    """
    Walk-forward (expanding window) one-step-ahead ARIMA forecasting.

    For each day in the test set:
      - Fit ARIMA on ALL data up to that day (no look-ahead)
      - Forecast the next day's return
      - Record prediction vs actual

    This is the correct way to evaluate time-series models and avoids
    information leakage from future data.
    """
    n_test  = int(len(returns) * TEST_RATIO)
    n_train = len(returns) - n_test

    print(f"Walk-forward ARIMA{ARIMA_ORDER}")
    print(f"Train: {n_train:,} days  |  Test: {n_test:,} days")
    print("Running walk-forward evaluation (this may take 1–2 minutes)...\n")

    actuals    = []
    forecasts  = []

    for i in range(n_test):
        train_window = returns.iloc[:n_train + i]

        try:
            model  = ARIMA(train_window, order=ARIMA_ORDER)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1).iloc[0]
        except Exception:
            forecast = 0.0   # fallback on rare convergence failure

        actuals.append(returns.iloc[n_train + i])
        forecasts.append(forecast)

        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}/{n_test}...")

    print(f"  Done. {n_test} forecasts generated.\n")
    return np.array(actuals), np.array(forecasts)


# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
def directional_accuracy(y_true, y_pred) -> float:
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def evaluate(y_true, y_pred, model_name: str):
    da   = directional_accuracy(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    print(f"{'=' * 45}")
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
    print("  arima.py  —  ARIMA Numerical Baseline")
    print("=" * 50 + "\n")

    returns           = load_returns()
    actuals, forecasts = walk_forward_arima(returns)
    results           = evaluate(actuals, forecasts, f"ARIMA{ARIMA_ORDER} (Numerical Only)")

    print("Interpretation:")
    print("  ARIMA models temporal autocorrelation in returns.")
    print("  Near-zero predictions are expected — returns are mostly noise.")
    print("  Compare these metrics to ARIMAX (arima + sentiment) to see")
    print("  whether sentiment provides additional predictive value.\n")

    return results


if __name__ == "__main__":
    main()