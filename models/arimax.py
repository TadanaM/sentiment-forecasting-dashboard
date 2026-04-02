# =============================================================================
# arimax.py
# PURPOSE: ARIMAX hybrid model — ARIMA extended with sentiment as an
#          exogenous regressor.
#
# This model answers:
#   "Does sentiment explain information not already captured by price history?"
#
# The only difference from arima.py is the inclusion of the daily sentiment
# polarity index as an exogenous variable (exog parameter in ARIMA).
# All other settings, date ranges, and evaluation metrics are identical,
# ensuring a fair, controlled comparison.
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
ARIMA_ORDER = (1, 0, 1)
TEST_RATIO  = 0.2

DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
PRICES_PATH    = os.path.join(DATA_DIR, "AAPL_prices.csv")
SENTIMENT_PATH = os.path.join(DATA_DIR, "daily_sentiment_index.csv")


# ── Step 1: Load & align data ─────────────────────────────────────────────────
def load_aligned_data() -> pd.DataFrame:
    """
    Load AAPL returns and daily sentiment index, then inner-join on date
    so that both series are perfectly aligned to the same trading days.
    """
    prices    = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] for col in prices.columns]

    returns   = prices["Close"].pct_change().rename("return")

    sentiment = pd.read_csv(SENTIMENT_PATH, index_col=0, parse_dates=True)
    sentiment = sentiment[["polarity"]]

    df = pd.concat([returns, sentiment], axis=1, join="inner").dropna()

    print(f"Aligned dataset: {len(df):,} trading days  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"Columns: {list(df.columns)}\n")
    return df


# ── Step 2: Walk-forward ARIMAX ───────────────────────────────────────────────
def walk_forward_arimax(df: pd.DataFrame) -> tuple:
    """
    Walk-forward one-step-ahead ARIMAX forecasting.
    Sentiment polarity is passed as exogenous variable at each step.
    """
    n_test  = int(len(df) * TEST_RATIO)
    n_train = len(df) - n_test

    returns   = df["return"]
    sentiment = df["polarity"]

    print(f"Walk-forward ARIMAX{ARIMA_ORDER}  (with sentiment exog)")
    print(f"Train: {n_train:,} days  |  Test: {n_test:,} days")
    print("Running walk-forward evaluation...\n")

    actuals   = []
    forecasts = []

    for i in range(n_test):
        train_returns   = returns.iloc[:n_train + i]
        train_sentiment = sentiment.iloc[:n_train + i].values.reshape(-1, 1)
        next_sentiment  = sentiment.iloc[n_train + i]

        try:
            model  = ARIMA(train_returns,
                           order=ARIMA_ORDER,
                           exog=train_sentiment)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1,
                                       exog=np.array([[next_sentiment]])).iloc[0]
        except Exception:
            forecast = 0.0

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
    print("  arimax.py  —  ARIMAX Hybrid Model")
    print("=" * 50 + "\n")

    df                 = load_aligned_data()
    actuals, forecasts = walk_forward_arimax(df)
    results            = evaluate(actuals, forecasts,
                                  f"ARIMAX{ARIMA_ORDER} (ARIMA + Sentiment)")

    print("Interpretation:")
    print("  Compare these results to arima.py (numerical only).")
    print("  If DA / RMSE / MAE improve → sentiment adds predictive value.")
    print("  If unchanged → sentiment does not improve time-series forecasting.\n")

    return results


if __name__ == "__main__":
    main()