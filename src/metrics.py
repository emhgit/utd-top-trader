import numpy as np
import pandas as pd

def calculate_momentum_metrics(
    df: pd.DataFrame,
    benchmark_ticker: str = "SPY"
) -> pd.DataFrame:

    if len(df) < 40:
        raise ValueError("Need at least 40 rows of historical data.")

    # --- Returns ---
    daily_returns = df.pct_change()

    return_5d = df.pct_change(5)
    return_10d = df.pct_change(10)
    return_20d = df.pct_change(20)

    # --- Volatility-adjusted momentum ---
    vol_20d = daily_returns.rolling(20).std()

    risk_adj_20d = return_20d / vol_20d

    # --- Relative strength ---
    if benchmark_ticker not in df.columns:
        raise ValueError(
            f"Benchmark ticker '{benchmark_ticker}' not found."
        )

    spy_return_20d = return_20d[benchmark_ticker]

    rel_strength_20d = return_20d.sub(
        spy_return_20d,
        axis=0
    )

    latest_date = df.index[-1]

    screener_df = pd.DataFrame(index=df.columns)

    screener_df["ret_5d"] = return_5d.loc[latest_date]
    screener_df["ret_10d"] = return_10d.loc[latest_date]
    screener_df["ret_20d"] = return_20d.loc[latest_date]

    screener_df["risk_adj_20d"] = risk_adj_20d.loc[latest_date]

    screener_df["rel_strength_20d"] = rel_strength_20d.loc[latest_date]

    # Remove benchmark from buy universe
    screener_df = screener_df.drop(
        index=benchmark_ticker,
        errors="ignore"
    )

    # Remove NaNs before ranking
    screener_df = screener_df.dropna()

    # --- Percentile ranks ---
    screener_df["rank_20d"] = (
        screener_df["ret_20d"].rank(pct=True)
    )

    screener_df["rank_10d"] = (
        screener_df["ret_10d"].rank(pct=True)
    )

    screener_df["rank_rel"] = (
        screener_df["rel_strength_20d"].rank(pct=True)
    )

    screener_df["rank_risk_adj"] = (
        screener_df["risk_adj_20d"].rank(pct=True)
    )

    # --- Composite model ---
    screener_df["composite_score"] = (
        0.4 * screener_df["rank_20d"] +
        0.2 * screener_df["rank_10d"] +
        0.2 * screener_df["rank_rel"] +
        0.2 * screener_df["rank_risk_adj"]
    )

    return screener_df.sort_values(
        by="composite_score",
        ascending=False
    )