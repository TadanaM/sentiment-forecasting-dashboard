# =============================================================================
# hybrid_regression.py
# PURPOSE: Hybrid Linear Regression — lagged returns + sentiment polarity.
#
# This model answers:
#   "Does sentiment improve feature-based ML predictions of daily returns?"
#
# It is structurally identical to stocktest.py (Linear Regression baseline)
# with one addition: the daily sentiment polarity score is included as an
# extra input feature alongside lagged returns.
#
# By comparing hybrid_regression.py vs stocktest.py, we isolate the
# contribution of sentiment to the ML forecasting paradigm, complementing
# the ARIMA vs ARIMAX comparison in the time-series paradigm.
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ───────────────────────────────────────────────────────────────────
TEST_RATIO  = 0.2
N_LAGS      = 5

DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
PRICES_PATH    = os.path.join(DATA_DIR, "AAPL_prices.csv")
SENTIMENT_PATH = os.path.join(DATA_DIR, "daily_sentiment_index.csv")


# ── Step 1: Load & align data ─────────────────────────────────────────────────
def load_aligned_data() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] for col in prices.columns]

    df = prices[["Close"]].copy()
    df["return"] = df["Close"].pct_change()

    # Lagged return features
    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)

    # Target
    df["target"] = df["return"].shift(-1)

    # Merge sentiment
    sentiment = pd.read_csv(SENTIMENT_PATH, index_col=0, parse_dates=True)
    df = df.join(sentiment[["polarity"]], how="inner")
    df = df.dropna()

    print(f"Aligned dataset: {len(df):,} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"Features: lagged returns (×{N_LAGS}) + sentiment polarity\n")
    return df


# ── Step 2: Train / test split ────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    lag_cols     = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    feature_cols = lag_cols + ["polarity"]

    X = df[feature_cols].values
    y = df["target"].values

    split = int(len(X) * (1 - TEST_RATIO))
    return X[:split], X[split:], y[:split], y[split:]


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
    print("  hybrid_regression.py  —  Regression + Sentiment")
    print("=" * 50 + "\n")

    df = load_aligned_data()

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples\n")

    model  = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = evaluate(y_test, y_pred,
                       "Hybrid Regression (Numerical + Sentiment)")

    print("Interpretation:")
    print("  Compare these results to stocktest.py (numerical only).")
    print("  Sentiment coefficient shows how strongly polarity influences")
    print("  the predicted return within the linear framework.\n")

    # Show sentiment feature coefficient
    lag_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    feature_names = lag_cols + ["polarity"]
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    })
    print("Feature Coefficients:")
    print(coef_df.to_string(index=False))
    print()

    return results


if __name__ == "__main__":
    main()