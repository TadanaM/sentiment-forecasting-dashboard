# =============================================================================
# random_forest_sentiment.py
# PURPOSE: Random Forest hybrid model — lagged returns + sentiment polarity.
#
# Identical to random_forest_numerical.py in every way except one:
# sentiment polarity is added as an additional input feature.
#
# This tests whether a non-linear model can extract value from sentiment
# that linear models cannot. It is the most powerful model in the pipeline.
# If even Random Forest cannot benefit from sentiment, it strongly supports
# the conclusion that sentiment provides limited predictive value at daily
# frequency for this asset.
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ───────────────────────────────────────────────────────────────────
TEST_RATIO   = 0.2
N_LAGS       = 5
N_ESTIMATORS = 100
RANDOM_STATE = 42

DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
PRICES_PATH    = os.path.join(DATA_DIR, "AAPL_prices.csv")
SENTIMENT_PATH = os.path.join(DATA_DIR, "daily_sentiment_index.csv")


# ── Data preparation ──────────────────────────────────────────────────────────
def prepare_data() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] for col in prices.columns]

    df = prices[["Close"]].copy()
    df["return"] = df["Close"].pct_change()

    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)

    df["target"] = df["return"].shift(-1)

    sentiment = pd.read_csv(SENTIMENT_PATH, index_col=0, parse_dates=True)
    df = df.join(sentiment[["polarity"]], how="inner")
    df = df.dropna()

    print(f"Dataset: {len(df):,} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"Features: lagged returns (×{N_LAGS}) + sentiment polarity\n")
    return df


def split_data(df: pd.DataFrame):
    lag_cols     = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    feature_cols = lag_cols + ["polarity"]
    X = df[feature_cols].values
    y = df["target"].values
    split = int(len(X) * (1 - TEST_RATIO))
    return X[:split], X[split:], y[:split], y[split:], feature_cols


# ── Evaluate ──────────────────────────────────────────────────────────────────
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
    print("=" * 55)
    print("  random_forest_sentiment.py  —  RF + Sentiment")
    print("=" * 55 + "\n")

    df = prepare_data()
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                  random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = evaluate(y_test, y_pred,
                       "Random Forest (Numerical + Sentiment)")

    # Feature importance — shows whether RF uses sentiment at all
    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("Feature Importances (polarity rank reveals sentiment weight):")
    print(imp_df.to_string(index=False))
    print()

    return results


if __name__ == "__main__":
    main()