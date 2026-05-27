import yfinance as yf
import pandas as pd
from typing import List

def fetch_etf_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches adjusted closing price history for a specific list of ETFs.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Check for 'Adj Close', fallback to 'Close' if unavailable
    if 'Adj Close' in data.columns.levels[0]:
        df_subset = data['Adj Close']
    elif 'Close' in data.columns.levels[0]:
        df_subset = data['Close']
    else:
        df_subset = data

    # If it still has a MultiIndex, flatten it cleanly here
    if isinstance(df_subset.columns, pd.MultiIndex):
        df_subset.columns = df_subset.columns.get_level_values(0)
        
    return df_subset