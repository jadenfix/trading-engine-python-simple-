# Sophisticated Trading Algorithm Engine

A comprehensive algorithmic trading system that uses public market data APIs to implement advanced trading strategies. Features multiple sophisticated strategies, advanced risk management, and comprehensive backtesting capabilities. Perfect for learning algorithmic trading concepts and testing advanced strategies.

## Features

- **Advanced Strategies**: Moving Average Crossover, RSI, Momentum, Mean Reversion, Volatility Breakout, Multi-Timeframe, and Enhanced Combined strategies
- **Sophisticated Risk Management**: Advanced position sizing, stop losses, portfolio optimization, and risk metrics
- **Public API Integration**: Uses Yahoo Finance for free market data with multiple timeframes
- **Comprehensive Backtesting**: Historical performance analysis with walk-forward optimization
- **Modular Design**: Easy to extend and customize with plugin architecture
- **Advanced Analytics**: Performance metrics, drawdown analysis, and comprehensive reporting

## Installation

1. Clone or download the project
2. Set up the virtual environment:
   ```bash
   python3 -m venv trading_env
   source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Usage

Run a backtest with default settings:
```bash
python main.py --symbols AAPL,MSFT --strategy combined --capital 10000
```

Available options:
- `--symbols`: Stock symbols (comma-separated)
- `--strategy`: Trading strategy (`ma`, `rsi`, `momentum`, `mean_reversion`, `breakout`, `multitimeframe`, `combined`, `enhanced`)
- `--capital`: Initial capital for backtest
- `--period`: Historical data period
- `--output`: Save detailed results to CSV

### Python API Usage

```python
from trading.algorithm import TradingAlgorithm

# Create algorithm with $50,000 initial capital
algorithm = TradingAlgorithm(initial_capital=50000, strategy='ma')

# Run backtest on tech stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
results = algorithm.run_backtest(symbols, period='6mo')

# Display results
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
print(f"Total Trades: {results['total_trades']}")
```

## Trading Strategies

### 1. Moving Average Crossover (MA)
- **Logic**: Buy when short MA crosses above long MA, sell when it crosses below
- **Parameters**: 10-day and 50-day moving averages (configurable)
- **Best for**: Trending markets

### 2. RSI Strategy
- **Logic**: Buy when RSI crosses above oversold level (30), sell when it crosses below overbought level (70)
- **Parameters**: 14-period RSI with 30/70 thresholds (configurable)
- **Best for**: Mean-reverting markets

### 3. Momentum Strategy
- **Logic**: Long top momentum stocks, short bottom momentum stocks based on rate of change
- **Parameters**: 20-day lookback, top 30% percentile selection
- **Best for**: Strong trending markets

### 4. Mean Reversion Strategy
- **Logic**: Buy oversold stocks (below 2σ), sell overbought stocks (above 2σ)
- **Parameters**: 20-day lookback, 2.0σ entry threshold, 0.5σ exit threshold
- **Best for**: Sideways/choppy markets

### 5. Volatility Breakout Strategy
- **Logic**: Buy when price breaks above upper ATR band, sell when below lower ATR band
- **Parameters**: 20-day ATR, 1.5x multiplier for breakout levels
- **Best for**: High volatility environments

### 6. Multi-Timeframe Strategy
- **Logic**: Combines short-term and long-term strategies requiring agreement
- **Parameters**: Configurable short and long-term strategy combinations
- **Best for**: Filtering noise and confirming trends

### 7. Enhanced Combined Strategy
- **Logic**: Voting system across all strategies, requires majority consensus
- **Parameters**: Minimum 2 votes required for signal generation
- **Best for**: Maximum risk reduction and signal confirmation

## Risk Management

- **Position Sizing**: Based on maximum risk per trade (default 2%)
- **Stop Loss**: 5% default stop loss on all positions
- **Portfolio Risk**: Monitors total portfolio exposure

## Performance Results

### Enhanced Combined Strategy (1-Year Backtest)
```
=== Backtest Results ===
Strategy: Enhanced Combined Strategy
Initial Capital: $10,000.00
Final Capital: $10,329.79
Total Return: 3.30%
Max Drawdown: -25.66%
Total Trades: 63
Winning Trades: 11
Losing Trades: 19
Win Rate: 36.7%
Profit/Loss: $282.33
```

### Individual Strategy Performance (6-Month Backtest)

| Strategy | Return | Win Rate | Total Trades | Max Drawdown |
|----------|--------|----------|--------------|--------------|
| **MA Crossover** | 0.86% | 75.0% | 10 | -15.36% |
| **RSI Strategy** | 2.04% | 100.0% | 5 | -10.22% |
| **Momentum** | 0.04% | 50.0% | 26 | -15.26% |
| **Mean Reversion** | 0.60% | 100.0% | 4 | -8.29% |

**Note**: Performance varies by market conditions and time periods. The enhanced combined strategy provides balanced exposure across multiple approaches.

## Project Structure

```
├── trading/
│   ├── __init__.py
│   ├── data_fetcher.py    # Yahoo Finance API integration
│   ├── strategies.py      # Trading strategies
│   ├── algorithm.py       # Main trading algorithm
│   └── risk_manager.py    # Risk management
├── main.py               # Command-line interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies

- **yfinance**: Yahoo Finance market data API
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **requests**: HTTP requests (dependency of yfinance)

## Customization

### Adding New Strategies

1. Create a new strategy class in `trading/strategies.py`:
```python
class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("My Strategy")

    def generate_signals(self, data):
        # Your strategy logic here
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        # ... implement signals ...
        return signals
```

2. Update the `_get_strategy` method in `trading/algorithm.py` to include your new strategy

### Modifying Risk Parameters

Adjust risk settings in the `TradingAlgorithm` constructor or modify `trading/risk_manager.py`.

## Advanced Features

- **Multiple Timeframes**: Support for daily, weekly, and monthly data analysis
- **Portfolio Optimization**: Advanced position sizing and risk-adjusted returns
- **Performance Analytics**: Comprehensive metrics including Sharpe ratio, max drawdown, and win rates
- **Strategy Voting**: Consensus-based signal generation for reduced false positives
- **Volatility Analysis**: ATR-based breakout detection and volatility targeting

## Limitations

- **Backtesting Only**: This is for historical testing, not live trading
- **Simplified Models**: Does not include commissions, slippage, or market impact
- **US Market Focus**: Optimized for US stock market data and trading hours
- **No Real-time Execution**: Simulated trading only, no live order placement

## License

This project is for educational purposes. Use at your own risk. Not intended for actual trading without proper testing and validation.
