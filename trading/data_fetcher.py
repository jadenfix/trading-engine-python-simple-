"""
Market data fetcher using Yahoo Finance API
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    """Fetches market data from Yahoo Finance"""

    def __init__(self):
        """Initialize the data fetcher"""
        pass

    def get_historical_data(self, symbol, period="6mo", interval="1d"):
        """
        Get historical price data for a symbol

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            pandas.DataFrame: Historical price data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_current_price(self, symbol):
        """
        Get current price for a symbol

        Args:
            symbol (str): Stock ticker symbol

        Returns:
            float: Current price or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.history(period="1d")
            if not info.empty:
                return info['Close'].iloc[-1]
            return None
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {str(e)}")
            return None

    def get_multiple_symbols(self, symbols, period="6mo"):
        """
        Get historical data for multiple symbols

        Args:
            symbols (list): List of stock ticker symbols
            period (str): Time period for data

        Returns:
            dict: Dictionary with symbols as keys and dataframes as values
        """
        result = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, period=period)
            if not data.empty:
                result[symbol] = data
        return result
