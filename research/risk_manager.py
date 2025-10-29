"""
Advanced Risk Management System

This module implements sophisticated risk management techniques including:
- Conditional Value at Risk (CVaR)
- Drawdown control
- Kelly criterion optimization
- Portfolio stress testing
- Dynamic risk budgeting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import norm, t
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class RiskManager:
    """Advanced risk management system"""

    def __init__(self, confidence_level=0.95, risk_free_rate=0.02, max_drawdown_limit=0.20):
        """
        Initialize risk manager

        Args:
            confidence_level (float): Confidence level for VaR/CVaR calculations
            risk_free_rate (float): Risk-free rate
            max_drawdown_limit (float): Maximum allowed drawdown
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_limit = max_drawdown_limit

        # Risk metrics history
        self.risk_history = []
        self.portfolio_history = []

    def calculate_var(self, returns, method='historical', window=252):
        """
        Calculate Value at Risk

        Args:
            returns (pd.Series): Return series
            method (str): 'historical', 'parametric', or 'monte_carlo'
            window (int): Lookback window for calculation

        Returns:
            dict: VaR metrics
        """
        if len(returns) < window:
            return {'var': 0, 'expected_shortfall': 0, 'confidence_level': self.confidence_level}

        # Use recent returns
        recent_returns = returns.tail(window)

        if method == 'historical':
            # Historical VaR
            var = np.percentile(recent_returns, (1 - self.confidence_level) * 100)

        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mean = recent_returns.mean()
            std = recent_returns.std()
            var = mean + std * norm.ppf(1 - self.confidence_level)

        elif method == 'monte_carlo':
            # Monte Carlo VaR
            n_simulations = 10000
            simulated_returns = np.random.choice(recent_returns, size=n_simulations, replace=True)
            var = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)

        else:
            var = 0

        # Calculate Expected Shortfall (CVaR)
        tail_returns = recent_returns[recent_returns <= var]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var

        return {
            'var': var,
            'expected_shortfall': expected_shortfall,
            'confidence_level': self.confidence_level,
            'method': method
        }

    def calculate_kelly_criterion(self, returns, current_portfolio_value=100000):
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            returns (pd.Series): Historical returns
            current_portfolio_value (float): Current portfolio value

        Returns:
            dict: Kelly criterion results
        """
        if len(returns) < 30:
            return {'kelly_fraction': 0, 'optimal_position': 0, 'edge': 0, 'odds_ratio': 0}

        # Calculate win probability and win/loss ratio
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()
        total_trades = len(returns)

        if total_trades == 0:
            return {'kelly_fraction': 0, 'optimal_position': 0, 'edge': 0, 'odds_ratio': 0}

        win_probability = winning_trades / total_trades
        loss_probability = losing_trades / total_trades

        if loss_probability == 0:
            return {'kelly_fraction': 0, 'optimal_position': 0, 'edge': 0, 'odds_ratio': 0}

        # Average win and loss
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())

        if avg_loss == 0:
            return {'kelly_fraction': 0, 'optimal_position': 0, 'edge': 0, 'odds_ratio': 0}

        odds_ratio = avg_win / avg_loss

        # Kelly fraction
        kelly_fraction = (odds_ratio * win_probability - loss_probability) / odds_ratio

        # Constrain to reasonable bounds
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Max 50% of portfolio

        # Optimal position size
        optimal_position = kelly_fraction * current_portfolio_value

        # Edge (expected value per unit bet)
        edge = win_probability * avg_win - loss_probability * avg_loss

        return {
            'kelly_fraction': kelly_fraction,
            'optimal_position': optimal_position,
            'edge': edge,
            'odds_ratio': odds_ratio,
            'win_probability': win_probability,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def monitor_drawdown(self, portfolio_values):
        """
        Monitor portfolio drawdown

        Args:
            portfolio_values (pd.Series): Portfolio values over time

        Returns:
            dict: Drawdown analysis
        """
        if len(portfolio_values) < 2:
            return {'current_drawdown': 0, 'max_drawdown': 0, 'recovery_time': 0}

        # Calculate drawdown
        cumulative = portfolio_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        current_drawdown = drawdown.iloc[-1]
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()

        # Recovery time (days since max drawdown)
        if max_drawdown < 0:
            recovery_dates = portfolio_values[portfolio_values.index > max_drawdown_date]
            recovery_date = recovery_dates[recovery_dates >= running_max.loc[max_drawdown_date]].first_valid_index()

            if recovery_date is not None:
                recovery_time = (recovery_date - max_drawdown_date).days
            else:
                recovery_time = (portfolio_values.index[-1] - max_drawdown_date).days
        else:
            recovery_time = 0

        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'recovery_time': recovery_time,
            'is_in_drawdown': current_drawdown < -0.05  # 5% drawdown threshold
        }

    def stress_test_portfolio(self, portfolio_returns, scenarios=None):
        """
        Stress test portfolio under various scenarios

        Args:
            portfolio_returns (pd.Series): Historical portfolio returns
            scenarios (dict): Custom stress scenarios

        Returns:
            dict: Stress test results
        """
        if scenarios is None:
            # Default stress scenarios
            scenarios = {
                'market_crash': -0.15,  # 15% drop
                'flash_crash': -0.10,   # 10% drop
                'volatility_spike': portfolio_returns.std() * 3,  # 3-sigma event
                'liquidity_crisis': -0.20,  # 20% drop
                'recovery_scenario': 0.10   # 10% gain
            }

        stress_results = {}

        for scenario_name, shock in scenarios.items():
            if 'volatility' in scenario_name.lower():
                # Volatility scenario
                stressed_returns = portfolio_returns * shock / portfolio_returns.std()
            else:
                # Price shock scenario
                stressed_returns = portfolio_returns + shock

            # Calculate stressed portfolio value
            initial_value = 100000
            stressed_value = initial_value * (1 + stressed_returns).prod()

            # Calculate stressed metrics
            stressed_var = self.calculate_var(stressed_returns, method='historical')
            stressed_drawdown = self.monitor_drawdown(pd.Series([initial_value] + list(initial_value * (1 + stressed_returns).cumprod())))

            stress_results[scenario_name] = {
                'shock': shock,
                'final_value': stressed_value,
                'return': (stressed_value - initial_value) / initial_value,
                'var': stressed_var['var'],
                'max_drawdown': stressed_drawdown['max_drawdown'],
                'survival_probability': 1.0 if stressed_value > initial_value * 0.8 else 0.5  # Arbitrary survival threshold
            }

        return stress_results

    def dynamic_risk_budgeting(self, strategy_returns, target_volatility=0.15):
        """
        Dynamic risk budgeting for portfolio allocation

        Args:
            strategy_returns (dict): Returns for each strategy
            target_volatility (float): Target portfolio volatility

        Returns:
            dict: Risk-budgeted weights
        """
        if not strategy_returns:
            return {}

        # Calculate strategy volatilities and correlations
        strategy_vols = {}
        correlations = {}

        strategy_names = list(strategy_returns.keys())

        for strategy in strategy_names:
            returns = strategy_returns[strategy]
            if len(returns) > 30:
                strategy_vols[strategy] = returns.std()
            else:
                strategy_vols[strategy] = 0.20  # Default volatility

        # Calculate correlation matrix
        returns_df = pd.DataFrame(strategy_returns)
        corr_matrix = returns_df.corr()

        # Risk parity allocation
        weights = {}

        if strategy_vols:
            # Equal risk contribution
            total_risk = sum(strategy_vols.values())

            if total_risk > 0:
                for strategy in strategy_names:
                    vol = strategy_vols[strategy]
                    weights[strategy] = vol / total_risk

                    # Scale to target volatility
                    if target_volatility > 0:
                        portfolio_vol = np.sqrt(np.dot(np.dot(weights.values(), corr_matrix.values), weights.values()))
                        if portfolio_vol > 0:
                            scale_factor = target_volatility / portfolio_vol
                            weights[strategy] *= scale_factor

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def risk_adjusted_position_sizing(self, signal_strength, volatility, current_drawdown,
                                    max_position_size=0.1):
        """
        Risk-adjusted position sizing based on multiple factors

        Args:
            signal_strength (float): Strength of trading signal (0-1)
            volatility (float): Asset volatility
            current_drawdown (float): Current portfolio drawdown
            max_position_size (float): Maximum allowed position size

        Returns:
            float: Recommended position size
        """
        # Base position size from signal strength
        base_size = signal_strength * max_position_size

        # Volatility adjustment (lower volatility = larger position)
        vol_factor = 0.20 / max(volatility, 0.05)  # Target 20% volatility
        vol_factor = min(vol_factor, 3.0)  # Cap at 3x

        # Drawdown adjustment (higher drawdown = smaller position)
        if current_drawdown < -0.10:  # In significant drawdown
            drawdown_factor = 0.5  # Reduce position by half
        elif current_drawdown < -0.05:  # Moderate drawdown
            drawdown_factor = 0.75  # Reduce position by 25%
        else:
            drawdown_factor = 1.0  # No reduction

        # Kelly-inspired sizing
        kelly_size = min(signal_strength * 2 - 1, 0.5)  # Simplified Kelly
        kelly_size = max(kelly_size, 0)

        # Combine factors
        position_size = base_size * vol_factor * drawdown_factor * kelly_size

        # Final constraints
        position_size = min(position_size, max_position_size)
        position_size = max(position_size, 0.01)  # Minimum 1%

        return position_size

    def generate_risk_report(self, portfolio_history, strategy_returns=None):
        """
        Generate comprehensive risk report

        Args:
            portfolio_history (list): Portfolio value history
            strategy_returns (dict): Returns by strategy

        Returns:
            dict: Comprehensive risk report
        """
        if not portfolio_history:
            return {}

        # Extract portfolio values and returns
        values = [record['value'] for record in portfolio_history]
        dates = [record['date'] for record in portfolio_history]

        portfolio_series = pd.Series(values, index=dates)
        returns = portfolio_series.pct_change().dropna()

        # Core risk metrics
        var_metrics = self.calculate_var(returns)
        kelly_metrics = self.calculate_kelly_criterion(returns)
        drawdown_metrics = self.monitor_drawdown(portfolio_series)

        # Stress testing
        stress_results = self.stress_test_portfolio(returns)

        # Strategy-level risk if available
        strategy_risks = {}
        if strategy_returns:
            for strategy_name, strategy_rets in strategy_returns.items():
                if len(strategy_rets) > 30:
                    strategy_risks[strategy_name] = {
                        'volatility': strategy_rets.std(),
                        'sharpe': (strategy_rets.mean() - self.risk_free_rate/252) / strategy_rets.std() * np.sqrt(252),
                        'max_drawdown': self.monitor_drawdown(pd.Series(strategy_rets))['max_drawdown']
                    }

        # Risk budgeting
        risk_budget = {}
        if strategy_returns:
            risk_budget = self.dynamic_risk_budgeting(strategy_returns)

        report = {
            'portfolio_metrics': {
                'total_return': (values[-1] - values[0]) / values[0],
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252),
                'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
                'calmar_ratio': returns.mean() * 252 / abs(drawdown_metrics['max_drawdown']) if drawdown_metrics['max_drawdown'] != 0 else 0
            },
            'risk_metrics': {
                'value_at_risk': var_metrics['var'],
                'conditional_var': var_metrics['expected_shortfall'],
                'max_drawdown': drawdown_metrics['max_drawdown'],
                'current_drawdown': drawdown_metrics['current_drawdown'],
                'recovery_time': drawdown_metrics['recovery_time']
            },
            'position_sizing': {
                'kelly_fraction': kelly_metrics['kelly_fraction'],
                'optimal_position': kelly_metrics['optimal_position'],
                'edge': kelly_metrics['edge']
            },
            'stress_testing': stress_results,
            'strategy_risks': strategy_risks,
            'risk_budgeting': risk_budget,
            'risk_limits': {
                'max_drawdown_limit': self.max_drawdown_limit,
                'confidence_level': self.confidence_level,
                'risk_free_rate': self.risk_free_rate
            }
        }

        return report


class RiskControlOverlay:
    """Real-time risk control overlay for trading systems"""

    def __init__(self, risk_manager):
        """
        Initialize risk control overlay

        Args:
            risk_manager (RiskManager): Risk manager instance
        """
        self.risk_manager = risk_manager
        self.risk_breaches = []
        self.position_limits = {}

    def check_risk_limits(self, current_portfolio_value, current_positions, market_data):
        """
        Check if any risk limits are breached

        Args:
            current_portfolio_value (float): Current portfolio value
            current_positions (dict): Current positions
            market_data (dict): Current market data

        Returns:
            dict: Risk limit check results
        """
        breaches = []

        # Drawdown limit
        portfolio_history = [record['value'] for record in self.risk_manager.portfolio_history[-252:]]  # Last year
        if portfolio_history:
            portfolio_series = pd.Series(portfolio_history)
            drawdown = self.risk_manager.monitor_drawdown(portfolio_series)

            if abs(drawdown['max_drawdown']) > self.risk_manager.max_drawdown_limit:
                breaches.append({
                    'type': 'drawdown_limit',
                    'current': abs(drawdown['max_drawdown']),
                    'limit': self.risk_manager.max_drawdown_limit,
                    'severity': 'high'
                })

        # Position concentration limits
        for symbol, shares in current_positions.items():
            if symbol in market_data:
                position_value = abs(shares) * market_data[symbol]
                position_pct = position_value / current_portfolio_value

                if position_pct > 0.1:  # 10% concentration limit
                    breaches.append({
                        'type': 'concentration_limit',
                        'symbol': symbol,
                        'current': position_pct,
                        'limit': 0.1,
                        'severity': 'medium'
                    })

        # Portfolio volatility limit
        recent_returns = pd.Series([record['value'] for record in self.risk_manager.portfolio_history[-60:]])
        if len(recent_returns) > 30:
            returns = recent_returns.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            if volatility > 0.30:  # 30% volatility limit
                breaches.append({
                    'type': 'volatility_limit',
                    'current': volatility,
                    'limit': 0.30,
                    'severity': 'medium'
                })

        # Record breaches
        for breach in breaches:
            self.risk_breaches.append({
                'timestamp': datetime.now(),
                'breach': breach
            })

        return {
            'breaches': breaches,
            'breach_count': len(breaches),
            'critical_breaches': len([b for b in breaches if b['severity'] == 'high'])
        }

    def adjust_position_sizes(self, original_signals, risk_check_results):
        """
        Adjust position sizes based on risk limits

        Args:
            original_signals (dict): Original trading signals
            risk_check_results (dict): Risk limit check results

        Returns:
            dict: Risk-adjusted signals
        """
        adjusted_signals = original_signals.copy()

        # Reduce position sizes if risk limits breached
        if risk_check_results['breach_count'] > 0:
            risk_multiplier = 0.5  # Reduce positions by 50% when limits breached

            for symbol in adjusted_signals:
                original_signal = adjusted_signals[symbol]
                if isinstance(original_signal, dict) and 'signal' in original_signal:
                    # DataFrame signal
                    adjusted_signals[symbol] = original_signal * risk_multiplier
                elif isinstance(original_signal, (int, float)):
                    # Direct signal
                    adjusted_signals[symbol] = original_signal * risk_multiplier

        return adjusted_signals


# Example usage
if __name__ == "__main__":
    print("Advanced Risk Management System")
    print("=" * 40)

    # Test risk manager
    risk_manager = RiskManager()

    # Generate sample returns
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns

    var_result = risk_manager.calculate_var(pd.Series(sample_returns))
    kelly_result = risk_manager.calculate_kelly_criterion(pd.Series(sample_returns))

    print(f"VaR (95%): {var_result['var']:.2%}")
    print(f"Expected Shortfall: {var_result['expected_shortfall']:.2%}")
    print(f"Kelly Fraction: {kelly_result['kelly_fraction']:.1%}")
    print(f"Edge: {kelly_result['edge']:.4f}")

    print("\nRisk management system ready!")
