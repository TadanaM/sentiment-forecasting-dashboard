# =============================================================================
# metrics.py
# PURPOSE: Run all models and produce a unified comparison table.
#
# This is the master evaluation script. It imports and runs every model,
# collects results, and prints a formatted comparison table showing:
#   - Directional Accuracy (DA)
#   - RMSE
#   - MAE
#
# for all numerical baselines and sentiment-augmented hybrid models side
# by side. This table forms the core of the Results section in the dissertation.
# =============================================================================

import sys
import os
import pandas as pd

# Add models directory to path so imports resolve correctly
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
sys.path.insert(0, os.path.abspath(MODELS_DIR))

# ── Run all models ────────────────────────────────────────────────────────────
def run_all_models() -> pd.DataFrame:
    results = []

    print("\n" + "█" * 55)
    print("  RUNNING ALL MODELS — FULL EVALUATION")
    print("█" * 55 + "\n")

    # ── Numerical Baselines ──
    print("━" * 55)
    print("  NUMERICAL BASELINES (no sentiment)")
    print("━" * 55)

    from stocktest import main as run_lr
    results.append(run_lr())

    from arima import main as run_arima
    results.append(run_arima())

    from random_forest_numerical import main as run_rf_num
    results.append(run_rf_num())

    # ── Hybrid Models ──
    print("━" * 55)
    print("  HYBRID MODELS (with sentiment)")
    print("━" * 55)

    from hybrid_regression import main as run_hybrid_lr
    results.append(run_hybrid_lr())

    from arimax import main as run_arimax
    results.append(run_arimax())

    from random_forest_sentiment import main as run_rf_sent
    results.append(run_rf_sent())

    return pd.DataFrame(results)


# ── Format comparison table ───────────────────────────────────────────────────
def print_comparison_table(df: pd.DataFrame):
    print("\n" + "═" * 65)
    print("  FULL MODEL COMPARISON TABLE")
    print("═" * 65)

    header = f"{'Model':<42} {'DA':>6} {'RMSE':>10} {'MAE':>10}"
    print(header)
    print("─" * 65)

    numerical_models = [
        "Linear Regression (Numerical Only)",
        "ARIMA(1, 0, 1) (Numerical Only)",
        "Random Forest (Numerical Only)",
    ]

    for _, row in df.iterrows():
        tag = "  [baseline]" if row["model"] in numerical_models else "  [hybrid]  "
        name = row["model"][:41]
        print(f"{name:<42} {row['DA']:>6.4f} {row['RMSE']:>10.6f} {row['MAE']:>10.6f}{tag}")

    print("═" * 65)

    # ── Delta analysis (hybrid vs its baseline) ──
    print("\n  SENTIMENT IMPACT (hybrid minus its numerical counterpart)")
    print("─" * 65)

    pairs = [
        ("Linear Regression (Numerical Only)",
         "Hybrid Regression (Numerical + Sentiment)"),
        ("ARIMA(1, 0, 1) (Numerical Only)",
         "ARIMAX(1, 0, 1) (ARIMA + Sentiment)"),
        ("Random Forest (Numerical Only)",
         "Random Forest (Numerical + Sentiment)"),
    ]

    for baseline_name, hybrid_name in pairs:
        base_row   = df[df["model"] == baseline_name]
        hybrid_row = df[df["model"] == hybrid_name]

        if base_row.empty or hybrid_row.empty:
            continue

        b = base_row.iloc[0]
        h = hybrid_row.iloc[0]

        delta_da   = h["DA"]   - b["DA"]
        delta_rmse = h["RMSE"] - b["RMSE"]
        delta_mae  = h["MAE"]  - b["MAE"]

        direction_da   = "↑" if delta_da   > 0 else ("↓" if delta_da   < 0 else "→")
        direction_rmse = "↓" if delta_rmse < 0 else ("↑" if delta_rmse > 0 else "→")
        direction_mae  = "↓" if delta_mae  < 0 else ("↑" if delta_mae  > 0 else "→")

        print(f"\n  {baseline_name.split('(')[0].strip()} vs Hybrid:")
        print(f"    DA   : {delta_da:+.4f} {direction_da}   "
              f"(+ve = sentiment improved direction prediction)")
        print(f"    RMSE : {delta_rmse:+.6f} {direction_rmse}   "
              f"(-ve = sentiment reduced error)")
        print(f"    MAE  : {delta_mae:+.6f} {direction_mae}   "
              f"(-ve = sentiment reduced error)")

    print("\n" + "═" * 65)
    print("  KEY:")
    print("  DA > 0.50 = better than random  |  DA baseline ≈ 0.53–0.54")
    print("  RMSE / MAE in return units       |  0.015 ≈ 1.5% per day")
    print("  Small deltas are expected — daily returns are dominated by noise")
    print("═" * 65 + "\n")


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame):
    out_path = os.path.join(os.path.dirname(__file__),
                            "..", "data", "results_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))   # ensure relative paths work
    results_df = run_all_models()
    print_comparison_table(results_df)
    save_results(results_df)