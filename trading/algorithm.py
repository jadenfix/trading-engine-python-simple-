"""
Main trading algorithm that coordinates data fetching, strategy execution, and risk management
"""
import pandas as pd
from datetime import datetime
from .data_fetcher import DataFetcher
from .strategies import (
    MovingAverageCrossover, RSIStrategy, CombinedStrategy,
    MomentumStrategy, MeanReversionStrategy, VolatilityBreakoutStrategy,
    MultiTimeframeStrategy, EnhancedCombinedStrategy
)
from .risk_manager import RiskManager


class TradingAlgorithm:
    """Main trading algorithm class"""

    def __init__(self, initial_capital=100000, strategy='combined'):
        """
        Initialize trading algorithm

        Args:
            initial_capital (float): Starting capital
            strategy (str): Strategy to use ('ma', 'rsi', 'momentum', 'mean_reversion',
                          'breakout', 'multitimeframe', 'combined', 'enhanced')
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_history = []
        self.trades = []

        # Initialize components
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager()

        # Initialize strategy
        self.strategy = self._get_strategy(strategy)

    def _get_strategy(self, strategy_name):
        """Get strategy instance based on name"""
        strategies = {
            'ma': MovingAverageCrossover(),
            'rsi': RSIStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': VolatilityBreakoutStrategy(),
            'multitimeframe': MultiTimeframeStrategy(),
            'combined': CombinedStrategy(),
            'enhanced': EnhancedCombinedStrategy(),
            'enhanced_combined': EnhancedCombinedStrategy()
        }
        return strategies.get(strategy_name.lower(), EnhancedCombinedStrategy())

    def run_backtest(self, symbols, start_date=None, end_date=None, period="1y"):
        """
        Run backtest for given symbols

        Args:
            symbols (list): List of stock symbols to trade
            start_date (str, optional): Start date for backtest
            end_date (str, optional): End date for backtest
            period (str): Period for data if dates not specified

        Returns:
            dict: Backtest results
        """
        print(f"Running backtest with {self.strategy.name} strategy...")
        print(f"Symbols: {symbols}")
        print(f"Initial Capital: ${self.capital:.2f}")

        # Get historical data for all symbols
        all_data = {}
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                all_data[symbol] = data
            else:
                print(f"Warning: No data available for {symbol}")

        if not all_data:
            return {"error": "No data available for any symbols"}

        # Find common date range
        start_date = max(data.index[0] for data in all_data.values())
        end_date = min(data.index[-1] for data in all_data.values())

        print(f"Backtest period: {start_date.date()} to {end_date.date()}")

        # Run simulation day by day
        current_date = start_date
        results = []

        while current_date <= end_date:
            # Get current prices for all symbols
            current_prices = {}
            for symbol, data in all_data.items():
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']

            # Execute trading logic for each symbol
            for symbol, data in all_data.items():
                if current_date in data.index:
                    self._process_symbol(symbol, data, current_date, current_prices)

            # Record portfolio state
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.portfolio_history.append({
                'date': current_date,
                'capital': self.capital,
                'portfolio_value': portfolio_value,
                'total_value': self.capital + portfolio_value
            })

            current_date += pd.Timedelta(days=1)

        # Generate final report
        return self._generate_report()

    def _process_symbol(self, symbol, data, current_date, current_prices):
        """Process trading logic for a single symbol"""
        # Get current data up to today
        current_data = data.loc[:current_date]

        if len(current_data) < 50:  # Need minimum data for strategies
            return

        # Generate trading signals
        signal_data = self.strategy.generate_signals(current_data)

        if current_date not in signal_data.index:
            return

        current_signal = signal_data.loc[current_date, 'signal']
        current_price = current_prices[symbol]

        # Execute trades based on signals
        if current_signal == 1:  # Buy signal
            self._execute_buy(symbol, current_price, current_data)
        elif current_signal == -1:  # Sell signal
            self._execute_sell(symbol, current_price, current_data)

        # Check for stop losses and take profits
        self._check_exit_conditions(symbol, current_price)

    def _execute_buy(self, symbol, current_price, data):
        """Execute buy order"""
        # Check if we already have a position
        if symbol in self.positions and self.positions[symbol] > 0:
            return  # Already long

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.capital, current_price
        )

        if position_size <= 0:
            return

        # Check if we have enough capital
        cost = position_size * current_price
        if cost > self.capital:
            position_size = int(self.capital / current_price)
            cost = position_size * current_price

        if position_size <= 0:
            return

        # Execute buy
        self.capital -= cost
        self.positions[symbol] = position_size

        # Record trade
        self.trades.append({
            'date': data.index[-1],
            'symbol': symbol,
            'side': 'buy',
            'price': current_price,
            'shares': position_size,
            'capital_before': self.capital + cost,
            'capital_after': self.capital
        })

        print(f"BUY {position_size} shares of {symbol} at ${current_price:.2f}")

    def _execute_sell(self, symbol, current_price, data):
        """Execute sell order"""
        # Check if we have position to sell
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return  # No position or already short

        position_size = self.positions[symbol]

        # Execute sell
        revenue = position_size * current_price
        self.capital += revenue
        self.positions[symbol] = 0

        # Record trade
        self.trades.append({
            'date': data.index[-1],
            'symbol': symbol,
            'side': 'sell',
            'price': current_price,
            'shares': position_size,
            'capital_before': self.capital - revenue,
            'capital_after': self.capital
        })

        print(f"SELL {position_size} shares of {symbol} at ${current_price:.2f}")

    def _check_exit_conditions(self, symbol, current_price):
        """Check for stop loss and take profit conditions"""
        if symbol not in self.positions or self.positions[symbol] == 0:
            return

        shares = self.positions[symbol]
        if shares == 0:
            return

        # This is simplified - in a real implementation, you'd need to track
        # entry price per position. For simplicity, we'll skip detailed exit logic
        # in the backtest and just rely on strategy signals

    def _calculate_portfolio_value(self, current_prices):
        """Calculate current portfolio value"""
        total_value = 0.0
        for symbol, shares in self.positions.items():
            if symbol in current_prices and shares > 0:
                total_value += shares * current_prices[symbol]
        return total_value

    def _generate_report(self):
        """Generate backtest performance report"""
        if not self.portfolio_history:
            return {"error": "No trading history"}

        # Convert to DataFrame for analysis
        history_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades)

        # Calculate performance metrics
        initial_value = self.initial_capital
        final_value = history_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100

        # Calculate drawdown
        history_df['peak'] = history_df['total_value'].cummax()
        history_df['drawdown'] = (history_df['total_value'] - history_df['peak']) / history_df['peak'] * 100
        max_drawdown = history_df['drawdown'].min()

        # Calculate trade statistics
        winning_trades = 0
        losing_trades = 0
        total_profit = 0

        if not trades_df.empty:
            for _, trade in trades_df.iterrows():
                if trade['side'] == 'buy':
                    # Find corresponding sell
                    sell_trades = trades_df[
                        (trades_df['symbol'] == trade['symbol']) &
                        (trades_df['side'] == 'sell') &
                        (trades_df['date'] > trade['date'])
                    ]
                    if not sell_trades.empty:
                        sell_trade = sell_trades.iloc[0]
                        profit = (sell_trade['price'] - trade['price']) * trade['shares']
                        total_profit += profit
                        if profit > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1

        win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0

        return {
            "strategy": self.strategy.name,
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "portfolio_history": history_df,
            "trades": trades_df
        }
