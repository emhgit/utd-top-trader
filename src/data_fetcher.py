import yfinance as yf
import pandas as pd
from typing import List

def fetch_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch adjusted close price history and return a clean price matrix.
    """

    print(f"Fetching data for {tickers} from {start_date} to {end_date}")

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    # yfinance may return either MultiIndex or flat columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df_subset = data["Close"]
        else:
            raise ValueError("Close prices not found in downloaded data.")
    else:
        df_subset = data.copy()

    # Remove completely empty tickers
    df_subset = df_subset.dropna(axis=1, how="all")

    # Force numeric dtype
    df_subset = df_subset.astype(float)

    return df_subset