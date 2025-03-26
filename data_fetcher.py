import yfinance as yf
import pandas as pd

class DataFetcher:
    """
    Responsible for fetching and preprocessing historical OHLCV data
    from Yahoo Finance for stocks.
    """
    def __init__(self):
        pass  
    
    def get_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given stock symbol and timeframe using Yahoo Finance.
        
        Parameters:
            symbol (str): Stock symbol (e.g., 'AAPL')
            interval (str): Data interval (e.g., '1d', '1h', '5m'). 
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
            pd.DataFrame: DataFrame with Date/Time as index and columns for 
            Open, High, Low, Close, Adj Close, and Volume.
        """
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            raise Exception("No data fetched from Yahoo Finance. Check the symbol, date range, or interval.")
        return df

    def get_quotes(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data and process it by selecting only the 'High', 'Low', and 'Close'
        columns.
        
        Parameters:
            symbol (str): Stock symbol 
            interval (str): Data interval
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
            pd.DataFrame: Processed DataFrame with Date/Time as index and columns for High, Low, and Close.
        """
        df = self.get_data(symbol, interval, start_date, end_date)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
     
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['High', 'Low', 'Close']]
        print("Data fetch successful.")
        return df


