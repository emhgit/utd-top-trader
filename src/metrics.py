import pandas as pd 

def calculate_momentum_metrics(df: pd.DataFrame, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    # 1. Calculate raw trailing returns
    return_5d = df.pct_change(5)
    return_10d = df.pct_change(10)
    return_20d = df.pct_change(20)

    # 2. Extract benchmark return to calculate Relative Strength (Alpha)
    # print(f"Available columns: {df.columns.tolist()}")
    
    if benchmark_ticker in df.columns:
        spy_return_20d = return_20d[benchmark_ticker]
        spy_strength_20d = return_20d.sub(spy_return_20d, axis=0)
    else: 
        raise ValueError(f"Benchmark ticker '{benchmark_ticker}' not found in DataFrame columns.")
    
    # 3. Take the latest available date row to generate today's screener data

    latest_date = df.index[-1]
    
    screener_df = pd.DataFrame(index=df.columns)
    screener_df['ret_5d'] = return_5d.loc[latest_date]
    screener_df['ret_10d'] = return_10d.loc[latest_date]
    screener_df['ret_20d'] = return_20d.loc[latest_date]
    screener_df['rel_strength_20d'] = spy_strength_20d.loc[latest_date]
    
    # Drop benchmark from selection pool as to not buy SPY itself
    screener_df = screener_df.drop(index = benchmark_ticker, errors='ignore')
    
    # 4. Generate Percentile Ranks (0.0 to 1.0) inside asset universe
    screener_df['rank_20d'] = screener_df['ret_20d'].rank(pct=True)
    screener_df['rank_10d'] = screener_df['ret_10d'].rank(pct=True)
    screener_df['rank_rel'] = screener_df['rel_strength_20d'].rank(pct=True)

    screener_df['composite_score'] = (
        0.5 * screener_df['rank_20d'] + 
        0.3 * screener_df['rank_10d'] + 
        0.2 * screener_df['rank_rel']
    )

    return screener_df.sort_values(by='composite_score', ascending=False)
