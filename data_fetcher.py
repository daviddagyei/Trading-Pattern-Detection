import requests
import pandas as pd
from datetime import datetime

class DataFetcher:
    """
    Responsible for fetching and preprocessing historical OHLCV data
    from CoinMarketCap API.
    """
    def __init__(self, api_key: str, proxies=None):
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.session = self._init_session(api_key, proxies)

    def _init_session(self, api_key, proxies):
        """
        Initialize and configure a requests.Session for making API calls.

        Args:
            api_key (str): CoinMarketCap API key.
            proxies (dict, optional): A dictionary of proxy settings.

        Returns:
            requests.Session: A configured session object.
        """
        session = requests.Session()
        session.headers.update({
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": api_key,
            "User-Agent": "coinmarketcap/python"
        })
        if proxies:
            session.proxies.update(proxies)
        return session

    def get_data(self, symbol: str, interval: str, start_date: str, end_date: str):
        """
        Fetch historical OHLCV data for a given symbol and timeframe.

        Parameters:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC')
            interval (str): Desired interval (e.g., '5min', '1h', '1d')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            dict: JSON response from the API.
        """
        endpoint = f"{self.base_url}/cryptocurrency/ohlcv/historical"
        params = {
            "symbol": symbol,
            "time_start": start_date,
            "time_end": end_date,
        }
        response = self.session.get(endpoint, params=params, timeout=30)
        if not response.ok:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        if "application/json" in response.headers.get("Content-Type", ""):
            return response.json()
        else:
            raise Exception("Unexpected response content type")
        
    def get_quotes(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Process API response from get_data to extract OHLCV quotes and return a pandas DataFrame.

        Parameters:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC')
            interval (str): Desired interval (e.g., '5min', '1h', '1d')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: DataFrame with Date/Time as index and columns for OHLCV.
        """
        json_response = self.get_data(symbol, interval, start_date, end_date)
        data_list = []
        for item in json_response.get("data", {}).get("quotes", []):
            dt = datetime.fromisoformat(item["time_open"])
            data_list.append({
                "Date": dt,
                "Open": float(item["quote"]["USD"]["open"]),
                "High": float(item["quote"]["USD"]["high"]),
                "Low": float(item["quote"]["USD"]["low"]),
                "Close": float(item["quote"]["USD"]["close"]),
                "Volume": float(item["quote"]["USD"]["volume"])
            })
        df = pd.DataFrame(data_list)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        print("success")
        return df
    
if __name__ == '__main__':
    API_KEY = "a8a0cc79-fab2-405e-83be-6661b0a126b8"
    
    fetcher = DataFetcher(API_KEY)
    
    symbol = "BTC"
    interval = "1d"  
    start_date = "2022-01-01"
    end_date = "2022-01-01" 
    
    df_quotes = fetcher.get_quotes(symbol, interval, start_date, end_date)
    
    print(df_quotes.head(10))
