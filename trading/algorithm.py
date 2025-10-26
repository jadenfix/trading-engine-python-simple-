"""
Main trading algorithm that coordinates data fetching, strategy execution, and risk management
"""
import pandas as pd
import numpy as np
from datetime import datetime
from .data_fetcher import DataFetcher
from .strategies import (
    MovingAverageCrossover, RSIStrategy, CombinedStrategy,
    MomentumStrategy, MeanReversionStrategy, VolatilityBreakoutStrategy,
    MultiTimeframeStrategy, EnhancedCombinedStrategy, ScalpingStrategy,
    ContrarianStrategy, LeveragedMomentumStrategy, MachineLearningStyleStrategy,
    ConservativeTrendStrategy, BalancedMultiStrategy, PairsTradingStrategy,
    StatisticalArbitrageStrategy, SectorRotationStrategy, MarketRegimeStrategy
)
from .risk_manager import RiskManager, AdvancedRiskManager


class TradingAlgorithm:
    """Main trading algorithm class"""

    def __init__(self, initial_capital=100000, strategy='enhanced', risk_profile='medium'):
        """
        Initialize trading algorithm

        Args:
            initial_capital (float): Starting capital
            strategy (str): Strategy to use ('ma', 'rsi', 'momentum', 'mean_reversion',
                          'breakout', 'multitimeframe', 'combined', 'enhanced', 'scalping',
                          'contrarian', 'leveraged_momentum', 'ml_style', 'conservative',
                          'balanced')
            risk_profile (str): Risk tolerance ('very_low', 'low', 'medium', 'high', 'very_high')
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_history = []
        self.trades = []

        # Initialize components
        self.data_fetcher = DataFetcher()
        self.risk_manager = AdvancedRiskManager(risk_profile=risk_profile)

        # Initialize strategy
        self.strategy = self._get_strategy(strategy)
        self.risk_profile = risk_profile

    def _get_strategy(self, strategy_name):
        """Get strategy instance based on name"""
        strategies = {
            # Basic strategies
            'ma': MovingAverageCrossover(),
            'rsi': RSIStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': VolatilityBreakoutStrategy(),

            # Advanced strategies
            'multitimeframe': MultiTimeframeStrategy(),
            'combined': CombinedStrategy(),
            'enhanced': EnhancedCombinedStrategy(),

            # Very High Risk strategies
            'scalping': ScalpingStrategy(),
            'contrarian': ContrarianStrategy(),
            'leveraged_momentum': LeveragedMomentumStrategy(),

            # Low Risk strategies
            'conservative': ConservativeTrendStrategy(),

            # Medium Risk strategies
            'balanced': BalancedMultiStrategy(),
            'ml_style': MachineLearningStyleStrategy(),

            # New Advanced strategies
            'pairs_trading': PairsTradingStrategy(),
            'statistical_arbitrage': StatisticalArbitrageStrategy(),
            'sector_rotation': SectorRotationStrategy(),
            'market_regime': MarketRegimeStrategy()
        }
        return strategies.get(strategy_name.lower(), EnhancedCombinedStrategy())

    def _is_multi_symbol_strategy(self):
        """Check if current strategy is a multi-symbol strategy"""
        multi_symbol_strategies = [
            'pairs_trading', 'statistical_arbitrage',
            'sector_rotation'
        ]
        return self.strategy.name.lower().replace(' ', '_') in multi_symbol_strategies

    def _process_multi_symbol(self, all_data, current_date, current_prices):
        """Process multi-symbol strategies"""
        # Get current data up to today for all symbols
        current_data_dict = {}
        for symbol, data in all_data.items():
            if current_date in data.index:
                current_data_dict[symbol] = data.loc[:current_date]

        if not current_data_dict:
            return

        # Generate signals for all symbols
        try:
            signals_dict = self.strategy.generate_signals(current_data_dict)

            # Process each symbol's signals
            for symbol, signal_data in signals_dict.items():
                if current_date in signal_data.index and symbol in current_prices:
                    current_signal = signal_data.loc[current_date, 'signal']
                    current_price = current_prices[symbol]

                    # Execute trades based on signals
                    if current_signal == 1:  # Buy signal
                        self._execute_buy(symbol, current_price, current_data_dict[symbol])
                    elif current_signal == -1:  # Sell signal
                        self._execute_sell(symbol, current_price, current_data_dict[symbol])

                    # Check for stop losses and take profits
                    self._check_exit_conditions(symbol, current_price)

        except Exception as e:
            print(f"Warning: Error processing multi-symbol strategy: {str(e)}")

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

        # Check if this is a multi-symbol strategy
        is_multi_symbol = self._is_multi_symbol_strategy()

        while current_date <= end_date:
            # Get current prices for all symbols
            current_prices = {}
            for symbol, data in all_data.items():
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']

            if is_multi_symbol:
                # Process all symbols together for multi-symbol strategies
                self._process_multi_symbol(all_data, current_date, current_prices)
            else:
                # Process each symbol individually for single-symbol strategies
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

        # Generate trading signals - handle both single and multi-symbol strategies
        if hasattr(self.strategy, 'generate_signals'):
            # Check if this is a multi-symbol strategy (returns dict) or single-symbol strategy (returns DataFrame)
            try:
                signal_result = self.strategy.generate_signals(current_data)

                # Multi-symbol strategies return dictionaries
                if isinstance(signal_result, dict):
                    # For multi-symbol strategies, we need to process all symbols together
                    # This is more complex - we'll need to modify the main loop
                    return

                # Single-symbol strategies return DataFrames
                signal_data = signal_result

            except Exception as e:
                print(f"Error in signal generation for {symbol}: {e}")
                print(f"Data type: {type(current_data)}")
                print(f"Data columns: {current_data.columns.tolist() if hasattr(current_data, 'columns') else 'No columns'}")
                import traceback
                traceback.print_exc()
                return

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

        # Calculate volatility for risk management
        try:
            volatility = data['Close'].pct_change().tail(20).std() if len(data) > 20 else 0.02
        except KeyError as e:
            print(f"KeyError in _execute_buy for {symbol}: {e}")
            print(f"Data type: {type(data)}")
            print(f"Data columns: {data.columns.tolist() if hasattr(data, 'columns') else 'No columns'}")
            return

        # Calculate position size with risk profile
        position_size = self.risk_manager.calculate_position_size(
            self.capital, current_price, volatility, symbol
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
            'capital_after': self.capital,
            'risk_profile': self.risk_profile
        })

        print(f"BUY {position_size} shares of {symbol} at ${current_price:.2f} (Risk: {self.risk_profile})")

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

        print(f"SELL {position_size} shares of {symbol} at ${current_price:.2f} (Risk: {self.risk_profile})")

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

    def _calculate_advanced_metrics(self, history_df, trades_df):
        """
        Calculate advanced performance metrics including Sharpe, Sortino, and Calmar ratios

        Args:
            history_df (pd.DataFrame): Portfolio history with daily values
            trades_df (pd.DataFrame): Individual trade data

        Returns:
            dict: Advanced performance metrics
        """
        if len(history_df) < 30:  # Need minimum data for meaningful metrics
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "information_ratio": 0.0,
                "treynor_ratio": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
                "volatility": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "recovery_factor": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0
            }

        # Calculate daily returns
        history_df = history_df.copy()
        history_df['daily_return'] = history_df['total_value'].pct_change()

        # Remove first row (NaN return)
        returns = history_df['daily_return'].dropna()

        # Risk-free rate (assume 2% annual)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate

        # Basic metrics
        total_return = (history_df['total_value'].iloc[-1] - history_df['total_value'].iloc[0]) / history_df['total_value'].iloc[0]
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Sharpe Ratio
        sharpe_ratio = (total_return - risk_free_rate * len(returns) / 252) / (volatility) if volatility > 0 else 0

        # Sortino Ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (total_return - risk_free_rate * len(returns) / 252) / (downside_volatility) if downside_volatility > 0 else 0

        # Calmar Ratio (return / max drawdown)
        max_drawdown = abs(history_df['drawdown'].min()) if 'drawdown' in history_df.columns else 0
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Value at Risk (95% confidence)
        var_95 = returns.quantile(0.05)

        # Conditional Value at Risk (expected loss beyond VaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

        # Recovery Factor (total return / max drawdown)
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

        # Profit Factor (gross profit / gross loss)
        winning_trades_profit = 0
        losing_trades_loss = 0

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
                        if profit > 0:
                            winning_trades_profit += profit
                        else:
                            losing_trades_loss -= profit

        profit_factor = winning_trades_profit / losing_trades_loss if losing_trades_loss > 0 else 0

        # Expectancy (average win * win rate - average loss * loss rate)
        if not trades_df.empty:
            # Calculate individual trade profits
            trade_profits = []
            for _, trade in trades_df.iterrows():
                if trade['side'] == 'buy':
                    sell_trades = trades_df[
                        (trades_df['symbol'] == trade['symbol']) &
                        (trades_df['side'] == 'sell') &
                        (trades_df['date'] > trade['date'])
                    ]
                    if not sell_trades.empty:
                        sell_trade = sell_trades.iloc[0]
                        profit = (sell_trade['price'] - trade['price']) * trade['shares']
                        trade_profits.append(profit)

            if trade_profits:
                avg_win = np.mean([p for p in trade_profits if p > 0]) if any(p > 0 for p in trade_profits) else 0
                avg_loss = abs(np.mean([p for p in trade_profits if p < 0])) if any(p < 0 for p in trade_profits) else 0
                win_rate = len([p for p in trade_profits if p > 0]) / len(trade_profits)
                loss_rate = 1 - win_rate
                expectancy = (avg_win * win_rate) - (avg_loss * loss_rate) if loss_rate > 0 else avg_win * win_rate
            else:
                expectancy = 0
        else:
            expectancy = 0

        # Information Ratio (excess return / tracking error)
        benchmark_return = risk_free_rate * len(returns)
        excess_return = total_return - benchmark_return
        tracking_error = returns.std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        # Treynor Ratio and Alpha (simplified)
        treynor_ratio = excess_return / 1.0 if True else 0
        alpha = excess_return - (risk_free_rate * len(returns) / 252)
        beta = 1.0

        return {
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "calmar_ratio": round(calmar_ratio, 4),
            "information_ratio": round(information_ratio, 4),
            "treynor_ratio": round(treynor_ratio, 4),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "volatility": round(volatility, 4),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            "recovery_factor": round(recovery_factor, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 2)
        }

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

        # Calculate advanced performance metrics
        advanced_metrics = self._calculate_advanced_metrics(history_df, trades_df)

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
            "trades": trades_df,
            **advanced_metrics  # Include advanced metrics
        }
