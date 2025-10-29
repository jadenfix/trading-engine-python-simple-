"""
Multi-Strategy Ensemble System with Regime-Dependent Allocation

This module implements an advanced ensemble system that dynamically allocates
across multiple quantitative strategies based on market regime detection,
correlation analysis, and risk management principles.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """Advanced market regime detection system"""

    def __init__(self, volatility_window=63, trend_window=252, correlation_window=100):
        """
        Initialize regime detector

        Args:
            volatility_window (int): Window for volatility regime detection
            trend_window (int): Window for trend regime detection
            correlation_window (int): Window for correlation regime detection
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.correlation_window = correlation_window

    def detect_market_regime(self, price_data_dict, current_date):
        """
        Detect current market regime across multiple dimensions

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Market regime classification
        """
        symbols = list(price_data_dict.keys())

        # Calculate market-wide metrics
        market_returns = []
        market_volatility = []

        for symbol in symbols:
            data = price_data_dict[symbol]
            if current_date in data.index and len(data) > self.volatility_window:
                recent_data = data.loc[:current_date].tail(self.volatility_window)
                returns = recent_data['Close'].pct_change().dropna()

                if len(returns) > 10:
                    market_returns.append(returns.mean())
                    market_volatility.append(returns.std())

        if not market_returns:
            return {'regime': 'insufficient_data', 'confidence': 0}

        # Aggregate market metrics
        avg_return = np.mean(market_returns)
        avg_volatility = np.mean(market_volatility)

        # Determine primary regime
        if avg_volatility > np.percentile(market_volatility, 75):
            volatility_regime = 'high_volatility'
        elif avg_volatility < np.percentile(market_volatility, 25):
            volatility_regime = 'low_volatility'
        else:
            volatility_regime = 'normal_volatility'

        if abs(avg_return) > avg_volatility * 2:
            trend_regime = 'strong_trend'
        elif abs(avg_return) > avg_volatility * 0.5:
            trend_regime = 'moderate_trend'
        else:
            trend_regime = 'sideways'

        # Combine regimes
        if volatility_regime == 'high_volatility' and trend_regime == 'strong_trend':
            primary_regime = 'crisis_trend'
        elif volatility_regime == 'high_volatility':
            primary_regime = 'high_volatility'
        elif trend_regime == 'strong_trend':
            primary_regime = 'strong_trend'
        elif volatility_regime == 'low_volatility' and trend_regime == 'sideways':
            primary_regime = 'low_volatility_sideways'
        else:
            primary_regime = 'normal_market'

        # Calculate regime confidence
        regime_confidence = min(1.0, (len(market_returns) / len(symbols)) * 0.8 + 0.2)

        return {
            'regime': primary_regime,
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'avg_return': avg_return,
            'avg_volatility': avg_volatility,
            'confidence': regime_confidence,
            'active_assets': len(market_returns)
        }


class StrategyEnsemble:
    """Multi-strategy ensemble with dynamic allocation"""

    def __init__(self, initial_weights=None, rebalance_frequency='daily',
                 max_strategy_weight=0.3, min_strategy_weight=0.05,
                 risk_parity=True):
        """
        Initialize strategy ensemble

        Args:
            initial_weights (dict): Initial strategy weights
            rebalance_frequency (str): How often to rebalance ('daily', 'weekly', 'monthly')
            max_strategy_weight (float): Maximum weight per strategy
            min_strategy_weight (float): Minimum weight per strategy
            risk_parity (bool): Whether to use risk parity allocation
        """
        self.initial_weights = initial_weights or {}
        self.rebalance_frequency = rebalance_frequency
        self.max_strategy_weight = max_strategy_weight
        self.min_strategy_weight = min_strategy_weight
        self.risk_parity = risk_parity

        # Strategy registry
        self.strategies = {}
        self.strategy_performance = {}
        self.current_weights = {}
        self.regime_history = []

        # Regime detector
        self.regime_detector = RegimeDetector()

    def register_strategy(self, name, strategy_class, parameters=None, category='default'):
        """
        Register a strategy in the ensemble

        Args:
            name (str): Strategy name
            strategy_class: Strategy class
            parameters (dict): Strategy parameters
            category (str): Strategy category for regime-based allocation
        """
        self.strategies[name] = {
            'class': strategy_class,
            'parameters': parameters or {},
            'category': category,
            'instance': None  # Will be instantiated when needed
        }

        if name not in self.current_weights:
            self.current_weights[name] = 1.0 / len(self.strategies)  # Equal weight initially

    def get_regime_based_weights(self, current_regime):
        """
        Calculate regime-based strategy weights

        Args:
            current_regime (dict): Current market regime

        Returns:
            dict: Strategy weights based on regime
        """
        regime = current_regime.get('regime', 'normal_market')

        # Define regime-specific strategy preferences
        regime_preferences = {
            'high_volatility': {
                'categories': ['mean_reversion', 'volatility', 'liquidity'],
                'avoid': ['momentum', 'trend_following']
            },
            'low_volatility_sideways': {
                'categories': ['mean_reversion', 'statistical_arbitrage', 'liquidity'],
                'avoid': ['momentum', 'breakout']
            },
            'strong_trend': {
                'categories': ['momentum', 'trend_following', 'breakout'],
                'avoid': ['mean_reversion', 'statistical_arbitrage']
            },
            'crisis_trend': {
                'categories': ['volatility', 'liquidity', 'crisis_alpha'],
                'avoid': ['momentum', 'trend_following']
            },
            'normal_market': {
                'categories': ['balanced', 'default'],
                'avoid': []
            }
        }

        preferences = regime_preferences.get(regime, regime_preferences['normal_market'])

        # Calculate weights based on preferences
        weights = {}
        preferred_strategies = []
        avoided_strategies = []

        for strategy_name, strategy_info in self.strategies.items():
            category = strategy_info['category']

            if category in preferences['categories']:
                preferred_strategies.append(strategy_name)
            elif category in preferences['avoid']:
                avoided_strategies.append(strategy_name)
            else:
                # Neutral strategies
                weights[strategy_name] = 0.1

        # Allocate higher weights to preferred strategies
        if preferred_strategies:
            preferred_weight = 0.7 / len(preferred_strategies)
            for strategy in preferred_strategies:
                weights[strategy] = preferred_weight

        # Reduce weights for avoided strategies
        if avoided_strategies:
            avoided_weight = 0.1 / len(avoided_strategies)
            for strategy in avoided_strategies:
                weights[strategy] = avoided_weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Apply constraints
        weights = self._apply_weight_constraints(weights)

        return weights

    def _apply_weight_constraints(self, weights):
        """
        Apply weight constraints to strategy allocation

        Args:
            weights (dict): Raw strategy weights

        Returns:
            dict: Constrained weights
        """
        # Apply maximum weight constraint
        for strategy in weights:
            weights[strategy] = min(weights[strategy], self.max_strategy_weight)

        # Apply minimum weight constraint and redistribute
        under_min = {k: v for k, v in weights.items() if v < self.min_strategy_weight}

        if under_min:
            # Redistribute weight from under-min strategies
            excess_weight = sum(self.min_strategy_weight - v for v in under_min.values())
            eligible_strategies = [k for k, v in weights.items() if v >= self.min_strategy_weight]

            if eligible_strategies:
                redistribution = excess_weight / len(eligible_strategies)
                for strategy in under_min:
                    weights[strategy] = self.min_strategy_weight

                for strategy in eligible_strategies:
                    weights[strategy] += redistribution

        # Final normalization
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def update_weights(self, current_regime, performance_data=None):
        """
        Update strategy weights based on regime and performance

        Args:
            current_regime (dict): Current market regime
            performance_data (dict): Recent performance data for strategies
        """
        # Get regime-based weights
        regime_weights = self.get_regime_based_weights(current_regime)

        # Apply risk parity if enabled
        if self.risk_parity and performance_data:
            risk_adjusted_weights = self._calculate_risk_parity_weights(regime_weights, performance_data)
        else:
            risk_adjusted_weights = regime_weights

        # Apply momentum adjustment based on recent performance
        if performance_data:
            momentum_weights = self._apply_performance_momentum(risk_adjusted_weights, performance_data)
        else:
            momentum_weights = risk_adjusted_weights

        self.current_weights = momentum_weights

    def _calculate_risk_parity_weights(self, base_weights, performance_data):
        """
        Calculate risk parity weights based on strategy volatilities

        Args:
            base_weights (dict): Base weights from regime analysis
            performance_data (dict): Performance data with volatility metrics

        Returns:
            dict: Risk parity adjusted weights
        """
        # Extract volatilities
        volatilities = {}
        for strategy_name in base_weights.keys():
            if strategy_name in performance_data:
                perf = performance_data[strategy_name]
                volatilities[strategy_name] = perf.get('volatility', 0.2)  # Default 20% vol

        if not volatilities:
            return base_weights

        # Risk parity: equal risk contribution
        # Weight = 1 / volatility, then normalize
        risk_weights = {}
        for strategy in volatilities:
            risk_weights[strategy] = 1.0 / max(volatilities[strategy], 0.01)  # Avoid division by zero

        # Normalize
        total_risk_weight = sum(risk_weights.values())
        risk_parity_weights = {k: v / total_risk_weight for k, v in risk_weights.items()}

        # Blend with base weights
        blended_weights = {}
        for strategy in base_weights:
            blended_weights[strategy] = 0.7 * risk_parity_weights.get(strategy, 0) + 0.3 * base_weights[strategy]

        return blended_weights

    def _apply_performance_momentum(self, base_weights, performance_data, momentum_window=20):
        """
        Apply performance momentum adjustment to weights

        Args:
            base_weights (dict): Base weights
            performance_data (dict): Performance data
            momentum_window (int): Lookback window for momentum

        Returns:
            dict: Momentum-adjusted weights
        """
        momentum_scores = {}

        for strategy_name in base_weights.keys():
            if strategy_name in performance_data:
                perf = performance_data[strategy_name]
                sharpe = perf.get('sharpe_ratio', 0)

                # Calculate momentum score based on recent Sharpe ratio
                momentum_scores[strategy_name] = max(0, sharpe)  # Only boost positive Sharpe

        if not momentum_scores:
            return base_weights

        # Apply momentum adjustment
        adjusted_weights = {}
        avg_momentum = np.mean(list(momentum_scores.values()))

        for strategy in base_weights:
            momentum_factor = momentum_scores.get(strategy, avg_momentum) / max(avg_momentum, 0.01)
            momentum_factor = min(max(momentum_factor, 0.5), 2.0)  # Constrain adjustment

            adjusted_weights[strategy] = base_weights[strategy] * momentum_factor

        # Re-normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def generate_ensemble_signals(self, price_data_dict, current_date):
        """
        Generate ensemble signals across all strategies

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Ensemble signals for each asset
        """
        # Detect current regime
        current_regime = self.regime_detector.detect_market_regime(price_data_dict, current_date)
        self.regime_history.append({
            'date': current_date,
            'regime': current_regime
        })

        # Update weights based on regime
        self.update_weights(current_regime)

        # Generate signals from each strategy
        strategy_signals = {}
        ensemble_signals = {}

        for strategy_name, strategy_info in self.strategies.items():
            try:
                # Instantiate strategy if needed
                if strategy_info['instance'] is None:
                    strategy_info['instance'] = strategy_info['class'](**strategy_info['parameters'])

                # Generate signals
                signals = strategy_info['instance'].generate_signals(price_data_dict, current_date)
                strategy_signals[strategy_name] = signals

            except Exception as e:
                print(f"Error generating signals for {strategy_name}: {e}")
                strategy_signals[strategy_name] = {}

        # Aggregate signals using current weights
        symbols = list(price_data_dict.keys())

        for symbol in symbols:
            symbol_signals = []

            for strategy_name, signals in strategy_signals.items():
                if symbol in signals and current_date in signals[symbol].index:
                    signal_value = signals[symbol].loc[current_date, 'signal']
                    weight = self.current_weights.get(strategy_name, 0)

                    if not pd.isna(signal_value):
                        symbol_signals.append(signal_value * weight)

            # Ensemble signal as weighted average
            if symbol_signals:
                ensemble_signal = np.mean(symbol_signals)

                # Convert to discrete signal
                if ensemble_signal > 0.3:
                    final_signal = 1
                elif ensemble_signal < -0.3:
                    final_signal = -1
                else:
                    final_signal = 0
            else:
                final_signal = 0

            # Create signal DataFrame
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = final_signal

                ensemble_signals[symbol] = signal_df

        return ensemble_signals

    def get_ensemble_summary(self):
        """
        Get summary of ensemble performance and allocation

        Returns:
            dict: Ensemble summary
        """
        return {
            'total_strategies': len(self.strategies),
            'current_weights': self.current_weights,
            'regime_history': self.regime_history[-10:] if self.regime_history else [],  # Last 10 regimes
            'strategy_categories': {name: info['category'] for name, info in self.strategies.items()}
        }


def create_unconventional_ensemble():
    """
    Create an ensemble with all unconventional strategies

    Returns:
        StrategyEnsemble: Configured ensemble
    """
    try:
        from .unconventional_strategies import (
            AttentionDrivenStrategy, SentimentRegimeStrategy, InformationTheoryStrategy,
            ComplexSystemsStrategy, FractalChaosStrategy, QuantumInspiredStrategy
        )
    except ImportError:
        # For standalone testing
        AttentionDrivenStrategy = None
        SentimentRegimeStrategy = None
        InformationTheoryStrategy = None
        ComplexSystemsStrategy = None
        FractalChaosStrategy = None
        QuantumInspiredStrategy = None

    # Create ensemble
    ensemble = StrategyEnsemble(
        rebalance_frequency='daily',
        max_strategy_weight=0.25,
        risk_parity=True
    )

    # Register strategies with appropriate categories
    if AttentionDrivenStrategy is not None:
        ensemble.register_strategy(
            'attention_driven',
            AttentionDrivenStrategy,
            {'attention_lookback': 21},
            'behavioral'
        )

        ensemble.register_strategy(
            'sentiment_regime',
            SentimentRegimeStrategy,
            {'sentiment_lookback': 63},
            'behavioral'
        )

        ensemble.register_strategy(
            'information_theory',
            InformationTheoryStrategy,
            {'entropy_window': 100},
            'complex_systems'
        )

        ensemble.register_strategy(
            'complex_systems',
            ComplexSystemsStrategy,
            {'network_lookback': 100},
            'complex_systems'
        )

        ensemble.register_strategy(
            'fractal_chaos',
            FractalChaosStrategy,
            {'fractal_window': 200},
            'chaos_theory'
        )

        ensemble.register_strategy(
            'quantum_inspired',
            QuantumInspiredStrategy,
            {'superposition_window': 50},
            'quantum'
        )
    else:
        # Register dummy strategies for testing
        class DummyStrategy:
            def __init__(self, **kwargs):
                pass
            def generate_signals(self, data, date):
                return {}

        ensemble.register_strategy('dummy_1', DummyStrategy, {}, 'default')
        ensemble.register_strategy('dummy_2', DummyStrategy, {}, 'default')

    return ensemble


# Example usage
if __name__ == "__main__":
    print("Strategy Ensemble System")
    print("=" * 40)

    # Create ensemble
    ensemble = create_unconventional_ensemble()

    summary = ensemble.get_ensemble_summary()

    print(f"Total Strategies: {summary['total_strategies']}")
    print(f"Strategy Categories: {summary['strategy_categories']}")
    print(f"Initial Weights: {summary['current_weights']}")

    print("\nEnsemble system initialized successfully!")
    print("Ready for regime-based multi-strategy allocation.")
