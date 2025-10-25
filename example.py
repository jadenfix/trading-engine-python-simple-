#!/usr/bin/env python3
"""
Example usage of the Simple Trading Algorithm
"""
from trading.algorithm import TradingAlgorithm
from trading.data_fetcher import DataFetcher
from trading.strategies import MovingAverageCrossover, RSIStrategy, CombinedStrategy


def example_basic_usage():
    """Basic example of using the trading algorithm"""
    print("=== Basic Trading Algorithm Example ===")

    # Create algorithm with different strategies
    strategies = {
        'Moving Average': TradingAlgorithm(initial_capital=10000, strategy='ma'),
        'RSI': TradingAlgorithm(initial_capital=10000, strategy='rsi'),
        'Combined': TradingAlgorithm(initial_capital=10000, strategy='combined')
    }

    # Test symbols
    symbols = ['AAPL', 'MSFT']

    for strategy_name, algorithm in strategies.items():
        print(f"\n--- {strategy_name} Strategy ---")

        # Run backtest
        results = algorithm.run_backtest(symbols, period='6mo')

        if 'error' not in results:
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Total Trades: {results['total_trades']}")


def example_custom_strategy():
    """Example of creating a custom strategy"""
    print("\n=== Custom Strategy Example ===")

    # You can create custom strategies by extending BaseStrategy
    # This is just an example - see strategies.py for implementation details

    custom_algorithm = TradingAlgorithm(initial_capital=50000, strategy='combined')

    # Test on more symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    results = custom_algorithm.run_backtest(symbols, period='1y')

    if 'error' not in results:
        print("Portfolio Performance:")
        print(f"Initial: ${results['initial_capital']:.2f}")
        print(f"Final: ${results['final_capital']:.2f}")
        print(f"Return: {results['total_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")

        # Show recent performance
        recent = results['portfolio_history'].tail(5)
        print("\nRecent Portfolio Values:")
        for _, row in recent.iterrows():
            print(f"  {row['date'].date()}: ${row['total_value']:.2f}")


def example_data_fetcher():
    """Example of using the data fetcher directly"""
    print("\n=== Data Fetcher Example ===")

    fetcher = DataFetcher()

    # Get current prices
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print("Current Prices:")
    for symbol in symbols:
        price = fetcher.get_current_price(symbol)
        if price:
            print(f"  {symbol}: ${price:.2f}")

    # Get historical data
    print("\nHistorical data for AAPL (last 5 days):")
    data = fetcher.get_historical_data('AAPL', period='5d')
    if not data.empty:
        print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())


if __name__ == "__main__":
    example_basic_usage()
    example_custom_strategy()
    example_data_fetcher()

    print("\n=== Example Complete ===")
    print("Check main.py for command-line usage")
    print("Modify this file to create your own trading experiments!")
