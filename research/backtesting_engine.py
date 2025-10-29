"""
Advanced Backtesting Engine for Research Strategies

This module provides a comprehensive backtesting framework for evaluating
quantitative trading strategies with realistic market conditions including
transaction costs, slippage, position sizing, and risk management.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BacktestingEngine:
    """Advanced backtesting engine with realistic market conditions"""

    def __init__(self, initial_capital=100000, commission_per_trade=0.001,
                 slippage_model='fixed', slippage_bps=5, max_position_size=0.1,
                 risk_free_rate=0.02):
        """
        Initialize backtesting engine

        Args:
            initial_capital (float): Starting capital
            commission_per_trade (float): Commission as fraction of trade value
            slippage_model (str): 'fixed', 'volume_based', or 'adaptive'
            slippage_bps (float): Slippage in basis points
            max_position_size (float): Maximum position size as fraction of capital
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate

        # Initialize portfolio state
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> shares
        self.trades = []  # List of trade records
        self.daily_returns = []
        self.portfolio_history = []

    def calculate_transaction_costs(self, symbol, shares, price, trade_type):
        """
        Calculate total transaction costs for a trade

        Args:
            symbol (str): Stock symbol
            shares (float): Number of shares
            price (float): Execution price
            trade_type (str): 'buy' or 'sell'

        Returns:
            float: Total transaction cost
        """
        trade_value = abs(shares) * price

        # Commission
        commission = trade_value * self.commission_per_trade

        # Slippage
        if self.slippage_model == 'fixed':
            slippage = trade_value * (self.slippage_bps / 10000)
        elif self.slippage_model == 'volume_based':
            # Simplified volume-based slippage (would need volume data)
            slippage = trade_value * (self.slippage_bps / 10000) * 1.5
        else:  # adaptive
            # Adaptive slippage based on position size
            position_pct = trade_value / self.portfolio_value
            slippage_multiplier = 1 + (position_pct / self.max_position_size)
            slippage = trade_value * (self.slippage_bps / 10000) * slippage_multiplier

        return commission + slippage

    def execute_trade(self, symbol, signal, current_price, current_date):
        """
        Execute a trade based on signal

        Args:
            symbol (str): Stock symbol
            signal (int): Trading signal (-1, 0, 1)
            current_price (float): Current market price
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Trade execution details
        """
        if signal == 0:
            return None

        # Determine trade direction and size
        current_position = self.positions.get(symbol, 0)

        if signal == 1:  # Buy signal
            if current_position < 0:  # Close short position and go long
                shares_to_trade = -current_position  # Close short
                trade_type = 'cover'
            else:  # Open or increase long position
                max_shares = int((self.portfolio_value * self.max_position_size) / current_price)
                target_position = max_shares
                shares_to_trade = target_position - current_position
                trade_type = 'buy' if shares_to_trade > 0 else None

        elif signal == -1:  # Sell signal
            if current_position > 0:  # Close long position and go short
                shares_to_trade = -current_position  # Close long
                trade_type = 'sell'
            else:  # Open or increase short position
                max_shares = int((self.portfolio_value * self.max_position_size) / current_price)
                target_position = -max_shares
                shares_to_trade = target_position - current_position
                trade_type = 'short' if shares_to_trade < 0 else None

        if trade_type is None or shares_to_trade == 0:
            return None

        # Calculate execution price with slippage
        if trade_type in ['buy', 'cover']:
            execution_price = current_price * (1 + self.slippage_bps / 10000)
        else:  # sell, short
            execution_price = current_price * (1 - self.slippage_bps / 10000)

        # Calculate transaction costs
        transaction_cost = self.calculate_transaction_costs(symbol, shares_to_trade, execution_price, trade_type)

        # Update positions and cash
        trade_value = abs(shares_to_trade) * execution_price
        total_cost = trade_value + transaction_cost

        if trade_type in ['buy', 'cover']:
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_trade
            else:
                return None  # Insufficient cash
        else:  # sell, short
            self.cash += trade_value - transaction_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_trade

        # Record trade
        trade_record = {
            'date': current_date,
            'symbol': symbol,
            'type': trade_type,
            'shares': shares_to_trade,
            'price': execution_price,
            'transaction_cost': transaction_cost,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash
        }

        self.trades.append(trade_record)
        return trade_record

    def update_portfolio_value(self, current_prices, current_date):
        """
        Update portfolio value based on current market prices

        Args:
            current_prices (dict): Current prices for all symbols
            current_date (pd.Timestamp): Current date
        """
        portfolio_value = self.cash

        for symbol, shares in self.positions.items():
            if symbol in current_prices and not pd.isna(current_prices[symbol]):
                portfolio_value += shares * current_prices[symbol]

        # Calculate daily return
        if self.portfolio_history:
            daily_return = (portfolio_value - self.portfolio_history[-1]['value']) / self.portfolio_history[-1]['value']
            self.daily_returns.append(daily_return)

        # Record portfolio state
        portfolio_record = {
            'date': current_date,
            'value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'total_positions': sum(abs(shares) for shares in self.positions.values())
        }

        self.portfolio_history.append(portfolio_record)
        self.portfolio_value = portfolio_value

    def run_backtest(self, signals_dict, price_data_dict, start_date=None, end_date=None):
        """
        Run backtest with signals and market data

        Args:
            signals_dict (dict): Dictionary of signal DataFrames by symbol
            price_data_dict (dict): Dictionary of price DataFrames by symbol
            start_date (pd.Timestamp): Start date for backtest
            end_date (pd.Timestamp): End date for backtest

        Returns:
            dict: Comprehensive backtest results
        """
        # Reset portfolio state
        self.__init__(self.initial_capital, self.commission_per_trade,
                      self.slippage_model, self.slippage_bps, self.max_position_size)

        # Determine date range
        all_dates = set()
        for data in price_data_dict.values():
            all_dates.update(data.index)

        if start_date is None:
            start_date = min(all_dates)
        if end_date is None:
            end_date = max(all_dates)

        backtest_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        # Run backtest
        for current_date in backtest_dates:
            # Get current prices
            current_prices = {}
            for symbol, data in price_data_dict.items():
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']

            # Execute trades based on signals
            for symbol, signal_df in signals_dict.items():
                if current_date in signal_df.index and symbol in current_prices:
                    signal = signal_df.loc[current_date, 'signal']
                    if not pd.isna(signal) and signal != 0:
                        self.execute_trade(symbol, int(signal), current_prices[symbol], current_date)

            # Update portfolio value
            self.update_portfolio_value(current_prices, current_date)

        # Calculate performance metrics
        results = self.calculate_performance_metrics()

        return results

    def calculate_performance_metrics(self):
        """
        Calculate comprehensive performance metrics

        Returns:
            dict: Performance metrics
        """
        if not self.portfolio_history:
            return {}

        # Basic returns
        portfolio_values = [record['value'] for record in self.portfolio_history]
        portfolio_returns = pd.Series(self.daily_returns)

        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital

        # Risk metrics
        if len(portfolio_returns) > 1:
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (portfolio_returns.mean() - self.risk_free_rate/252) / portfolio_returns.std() * np.sqrt(252)

            # Maximum drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

            # Win rate
            winning_trades = sum(1 for trade in self.trades if trade['type'] in ['sell', 'cover'])
            total_trades = len([trade for trade in self.trades if trade['type'] in ['buy', 'short']])
            win_rate = winning_trades / max(1, total_trades)

            # Profit factor
            gross_profits = sum(trade['price'] * abs(trade['shares']) for trade in self.trades
                              if trade['type'] in ['sell', 'cover'])
            gross_losses = sum(trade['price'] * abs(trade['shares']) for trade in self.trades
                             if trade['type'] in ['buy', 'short'])
            profit_factor = gross_profits / max(0.01, gross_losses)

        else:
            volatility = sharpe_ratio = max_drawdown = var_95 = cvar_95 = win_rate = profit_factor = 0

        # Transaction costs
        total_commission = sum(trade['transaction_cost'] for trade in self.trades)
        total_turnover = sum(abs(trade['shares']) * trade['price'] for trade in self.trades)

        return {
            'total_return': total_return,
            'annualized_return': total_return * (252 / max(1, len(portfolio_returns))),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'total_commission': total_commission,
            'turnover': total_turnover,
            'final_portfolio_value': portfolio_values[-1],
            'portfolio_history': self.portfolio_history,
            'trades': self.trades
        }


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameters"""

    def __init__(self, backtest_engine, optimization_window=252, validation_window=63):
        """
        Initialize walk-forward optimizer

        Args:
            backtest_engine (BacktestingEngine): Backtesting engine instance
            optimization_window (int): Window for parameter optimization (in days)
            validation_window (int): Window for parameter validation (in days)
        """
        self.backtest_engine = backtest_engine
        self.optimization_window = optimization_window
        self.validation_window = validation_window

    def optimize_parameters(self, strategy_class, parameter_ranges, signals_dict,
                          price_data_dict, start_date, end_date):
        """
        Optimize strategy parameters using walk-forward analysis

        Args:
            strategy_class: Strategy class to optimize
            parameter_ranges (dict): Parameter ranges for optimization
            signals_dict (dict): Signals dictionary
            price_data_dict (dict): Price data dictionary
            start_date (pd.Timestamp): Start date
            end_date (pd.Timestamp): End date

        Returns:
            dict: Optimization results
        """
        # This is a simplified implementation
        # In practice, you'd use grid search or Bayesian optimization

        best_params = {}
        best_sharpe = -np.inf

        # Simple grid search over parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)

        for params in param_combinations[:10]:  # Limit for demo
            try:
                # Create strategy with parameters
                strategy = strategy_class(**params)

                # Generate signals (simplified - would need to regenerate)
                test_signals = signals_dict

                # Run backtest
                results = self.backtest_engine.run_backtest(
                    test_signals, price_data_dict, start_date, end_date
                )

                if results.get('sharpe_ratio', -np.inf) > best_sharpe:
                    best_sharpe = results['sharpe_ratio']
                    best_params = params

            except Exception as e:
                continue

        return {
            'best_parameters': best_params,
            'best_sharpe_ratio': best_sharpe,
            'optimization_method': 'grid_search'
        }

    def _generate_parameter_combinations(self, parameter_ranges):
        """Generate parameter combinations for optimization"""
        # Simplified parameter combination generation
        combinations = []

        # Example for a simple 2-parameter strategy
        if 'lookback_period' in parameter_ranges and 'holding_period' in parameter_ranges:
            for lookback in parameter_ranges['lookback_period']:
                for holding in parameter_ranges['holding_period']:
                    combinations.append({
                        'lookback_period': lookback,
                        'holding_period': holding
                    })

        return combinations


def run_strategy_backtest(strategy_class, symbols, period='2y', strategy_params=None,
                         backtest_params=None):
    """
    Convenience function to run a complete strategy backtest

    Args:
        strategy_class: Strategy class to test
        symbols (list): List of symbols
        period (str): Data period
        strategy_params (dict): Strategy parameters
        backtest_params (dict): Backtest parameters

    Returns:
        dict: Complete backtest results
    """
    from trading.data_fetcher import DataFetcher
    from research.strategies import FactorMomentumStrategy  # Example import

    # Initialize components
    data_fetcher = DataFetcher()

    if backtest_params is None:
        backtest_params = {}

    backtest_engine = BacktestingEngine(**backtest_params)

    # Fetch data
    price_data = {}
    for symbol in symbols:
        data = data_fetcher.get_historical_data(symbol, period=period)
        if not data.empty:
            price_data[symbol] = data

    if not price_data:
        return {"error": "No data available"}

    # Initialize strategy
    if strategy_params is None:
        strategy_params = {}

    strategy = strategy_class(**strategy_params)

    # Generate signals for all dates
    signals_dict = {}
    all_dates = sorted(set().union(*[data.index for data in price_data.values()]))

    for current_date in all_dates:
        try:
            signals = strategy.generate_signals(price_data, current_date)
            for symbol, signal_df in signals.items():
                if symbol not in signals_dict:
                    signals_dict[symbol] = signal_df
        except:
            continue

    if not signals_dict:
        return {"error": "Failed to generate signals"}

    # Run backtest
    results = backtest_engine.run_backtest(signals_dict, price_data)

    return {
        'strategy_name': strategy.__class__.__name__,
        'symbols': symbols,
        'period': period,
        'strategy_params': strategy_params,
        'backtest_params': backtest_params,
        'performance': results
    }


# Example usage and testing
if __name__ == "__main__":
    # Test backtest engine with a simple example
    print("Backtesting Engine Test")
    print("=" * 30)

    # Simple test without external imports
    backtest_engine = BacktestingEngine(initial_capital=100000)

    print(f"Initial Capital: ${backtest_engine.initial_capital:,.0f}")
    print(f"Commission per Trade: {backtest_engine.commission_per_trade:.2%}")
    print(f"Slippage (bps): {backtest_engine.slippage_bps}")
    print("Backtesting Engine initialized successfully!")
