import numpy as np
import pandas as pd

TRADING_DAYS = 252

def calculate_core_performance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> dict:

    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()

    aligned = pd.concat(
        [strategy_returns, benchmark_returns],
        axis=1
    ).dropna()

    strategy_returns = aligned.iloc[:, 0]
    benchmark_returns = aligned.iloc[:, 1]

    # --- Total returns ---
    total_strat_ret = (
        (1 + strategy_returns).prod() - 1
    )

    total_bench_ret = (
        (1 + benchmark_returns).prod() - 1
    )

    # --- Annualized return ---
    strat_ann_return = (
        (1 + total_strat_ret)
        ** (TRADING_DAYS / len(strategy_returns))
        - 1
    )

    # --- Beta ---
    benchmark_variance = np.var(benchmark_returns)

    beta = (
        np.cov(strategy_returns, benchmark_returns)[0, 1]
        / benchmark_variance
        if benchmark_variance > 0
        else np.nan
    )

    # --- Alpha ---
    benchmark_ann_return = (
        (1 + total_bench_ret)
        ** (TRADING_DAYS / len(benchmark_returns))
        - 1
    )

    annualized_alpha = (
        strat_ann_return
        - beta * benchmark_ann_return
    )

    # --- Sharpe ratio ---
    strat_vol = (
        strategy_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        strat_ann_return / strat_vol
        if strat_vol > 0
        else np.nan
    )

    # --- Max drawdown ---
    equity_curve = (
        1 + strategy_returns
    ).cumprod()

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    return {
        "Strategy Return (%)":
            total_strat_ret * 100,

        "Benchmark Return (%)":
            total_bench_ret * 100,

        "Annualized Strategy Return (%)":
            strat_ann_return * 100,

        "Annualized Alpha (%)":
            annualized_alpha * 100,

        "Beta":
            beta,

        "Sharpe Ratio":
            sharpe,

        "Max Drawdown (%)":
            max_drawdown * 100
    }