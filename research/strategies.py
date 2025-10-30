"""
Emerging Quantitative Strategies for Statistical Edge Discovery

This module implements recently emerging quantitative strategies that are
statistically rigorous but not dependent on heavy machine learning.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from typing import Dict, List, Tuple, Optional
from trading.strategies import BaseStrategy


class FactorMomentumStrategy(BaseStrategy):
    """Factor momentum strategy based on cross-sectional factor analysis"""

    def __init__(self, lookback_period=252, formation_period=21, holding_period=21,
                 top_quantile=0.2, min_assets=5):
        """
        Initialize factor momentum strategy

        Args:
            lookback_period (int): Period for calculating factor exposures
            formation_period (int): Period for factor momentum calculation
            holding_period (int): Holding period for positions
            top_quantile (float): Top quantile for factor selection
            min_assets (int): Minimum assets required for strategy
        """
        super().__init__("Factor Momentum Strategy")
        self.lookback_period = lookback_period
        self.formation_period = formation_period
        self.holding_period = holding_period
        self.top_quantile = top_quantile
        self.min_assets = min_assets

    def calculate_factor_exposures(self, price_data_dict):
        """
        Calculate factor exposures for all assets

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values

        Returns:
            dict: Factor exposures for each asset
        """
        symbols = list(price_data_dict.keys())

        # Calculate returns
        returns_dict = {}
        for symbol, data in price_data_dict.items():
            returns_dict[symbol] = data['Close'].pct_change().dropna()

        # Find common dates
        common_dates = None
        for returns in returns_dict.values():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)

        if len(common_dates) < self.lookback_period:
            return {}

        # Align returns
        aligned_returns = {symbol: returns.loc[common_dates] for symbol, returns in returns_dict.items()}

        # Calculate factor exposures
        factor_exposures = {}

        for symbol in symbols:
            returns = aligned_returns[symbol]

            # Size factor (negative of momentum - smaller stocks tend to have higher volatility)
            size_exposure = -returns.rolling(self.lookback_period).mean()

            # Value factor (book-to-market proxy using volatility)
            value_exposure = returns.rolling(21).std() / returns.rolling(self.lookback_period).std()

            # Quality factor (Sharpe-like ratio)
            quality_exposure = returns.rolling(63).mean() / returns.rolling(self.lookback_period).std()

            # Momentum factor (price momentum)
            momentum_exposure = returns.rolling(self.lookback_period).mean()

            # Volatility factor (idiosyncratic volatility)
            market_returns = pd.DataFrame(aligned_returns).mean(axis=1)
            vol_exposure = returns.rolling(self.lookback_period).std()

            factor_exposures[symbol] = {
                'size': size_exposure,
                'value': value_exposure,
                'quality': quality_exposure,
                'momentum': momentum_exposure,
                'volatility': vol_exposure
            }

        return factor_exposures

    def calculate_factor_momentum(self, factor_exposures, formation_start, formation_end):
        """
        Calculate factor momentum over formation period

        Args:
            factor_exposures (dict): Factor exposures for each asset
            formation_start (pd.Timestamp): Start of formation period
            formation_end (pd.Timestamp): End of formation period

        Returns:
            dict: Factor momentum scores
        """
        factor_momentum = {}

        for factor_name in ['size', 'value', 'quality', 'momentum', 'volatility']:
            # Calculate cross-sectional average factor exposure
            factor_cross_section = []
            valid_dates = []

            for symbol, exposures in factor_exposures.items():
                if factor_name in exposures:
                    exposure_series = exposures[factor_name]
                    formation_exposure = exposure_series.loc[formation_start:formation_end]

                    if len(formation_exposure.dropna()) > 10:
                        factor_cross_section.append(formation_exposure)
                        valid_dates.extend(formation_exposure.dropna().index)

            if factor_cross_section:
                # Average factor exposure across assets
                avg_exposure = pd.concat(factor_cross_section, axis=1).mean(axis=1)
                avg_exposure = avg_exposure.loc[formation_start:formation_end]

                if len(avg_exposure) > self.formation_period // 2:
                    # Calculate factor momentum (trend in factor exposure)
                    factor_momentum[factor_name] = avg_exposure.rolling(self.formation_period // 4).mean()

        return factor_momentum

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate trading signals based on factor momentum

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        # Calculate factor exposures
        factor_exposures = self.calculate_factor_exposures(price_data_dict)

        if not factor_exposures:
            return {symbol: 0 for symbol in price_data_dict.keys()}

        # Define formation period
        formation_end = current_date
        formation_start = formation_end - pd.Timedelta(days=self.formation_period)

        # Calculate factor momentum
        factor_momentum = self.calculate_factor_momentum(factor_exposures, formation_start, formation_end)

        if not factor_momentum:
            return {symbol: 0 for symbol in price_data_dict.keys()}

        # Find strongest factor momentum
        best_factor = max(factor_momentum.items(), key=lambda x: abs(x[1].iloc[-1]) if len(x[1]) > 0 else 0)
        factor_name, momentum_series = best_factor

        if len(momentum_series) == 0 or abs(momentum_series.iloc[-1]) < 0.001:
            return {symbol: 0 for symbol in price_data_dict.keys()}

        # Select assets based on factor exposure to high-momentum factor
        signals = {}

        for symbol in price_data_dict.keys():
            if symbol in factor_exposures and factor_name in factor_exposures[symbol]:
                exposure_series = factor_exposures[symbol][factor_name]

                if len(exposure_series) > 30:
                    # Check if asset has positive exposure to strong factor
                    current_exposure = exposure_series.loc[:current_date].iloc[-1]

                    if not pd.isna(current_exposure):
                        # Long if positive exposure to positive momentum factor
                        # or negative exposure to negative momentum factor
                        if (momentum_series.iloc[-1] > 0 and current_exposure > 0) or \
                           (momentum_series.iloc[-1] < 0 and current_exposure < 0):
                            signals[symbol] = 1
                        else:
                            signals[symbol] = -1
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0
            else:
                signals[symbol] = 0

        # Convert signals to DataFrames as expected by the algorithm
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                # Create a DataFrame with the same index as the price data
                # Set signal to 0 for all dates except current_date where we set the actual signal
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class CrossSectionalMomentumStrategy(BaseStrategy):
    """Cross-sectional momentum strategy with statistical validation"""

    def __init__(self, lookback_period=252, holding_period=21, min_momentum=0.01,
                 volume_filter=True, volatility_threshold=0.02):
        """
        Initialize cross-sectional momentum strategy

        Args:
            lookback_period (int): Lookback period for momentum calculation
            holding_period (int): Holding period for positions
            min_momentum (float): Minimum momentum threshold
            volume_filter (bool): Whether to use volume confirmation
            volatility_threshold (float): Minimum volatility for trading
        """
        super().__init__("Cross-Sectional Momentum Strategy")
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.min_momentum = min_momentum
        self.volume_filter = volume_filter
        self.volatility_threshold = volatility_threshold

    def calculate_cross_sectional_ranks(self, price_data_dict, current_date):
        """
        Calculate cross-sectional momentum ranks

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for ranking

        Returns:
            dict: Momentum ranks for each asset
        """
        symbols = list(price_data_dict.keys())
        momentum_scores = {}

        for symbol in symbols:
            data = price_data_dict[symbol]

            if len(data) < self.lookback_period:
                momentum_scores[symbol] = 0
                continue

            # Calculate momentum
            current_price = data['Close'].loc[current_date]
            past_price = data['Close'].shift(self.lookback_period).loc[current_date]

            if pd.isna(current_price) or pd.isna(past_price) or past_price == 0:
                momentum_scores[symbol] = 0
                continue

            momentum = (current_price - past_price) / past_price

            # Volume confirmation
            if self.volume_filter:
                current_volume = data['Volume'].loc[current_date]
                avg_volume = data['Volume'].rolling(20).mean().loc[current_date]

                if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
                    momentum_scores[symbol] = 0
                    continue

                volume_ratio = current_volume / avg_volume
                if volume_ratio < 1.2:  # Require above-average volume
                    momentum_scores[symbol] = 0
                    continue

            # Volatility filter
            recent_volatility = data['Close'].pct_change().rolling(20).std().loc[current_date]
            if pd.isna(recent_volatility) or recent_volatility < self.volatility_threshold:
                momentum_scores[symbol] = 0
                continue

            momentum_scores[symbol] = momentum

        # Calculate cross-sectional ranks
        valid_scores = {symbol: score for symbol, score in momentum_scores.items() if score != 0}

        if not valid_scores:
            return {symbol: 0 for symbol in symbols}

        # Normalize scores and create ranks
        scores_df = pd.Series(valid_scores)
        ranks = (scores_df - scores_df.min()) / (scores_df.max() - scores_df.min())

        # Create final signal dictionary
        signals = {}
        for symbol in symbols:
            if symbol in valid_scores:
                score = valid_scores[symbol]
                rank = ranks[symbol]

                if score > self.min_momentum and rank > 0.7:  # Top 30% of performers
                    signals[symbol] = 1
                elif score < -self.min_momentum and rank < 0.3:  # Bottom 30% of performers
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0
            else:
                signals[symbol] = 0

        # Convert signals to DataFrames as expected by the algorithm
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                # Create a DataFrame with the same index as the price data
                # Set signal to 0 for all dates except current_date where we set the actual signal
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate trading signals based on cross-sectional momentum

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        # Calculate cross-sectional ranks
        momentum_ranks = self.calculate_cross_sectional_ranks(price_data_dict, current_date)

        if not momentum_ranks:
            return {symbol: 0 for symbol in price_data_dict.keys()}

        # Sort by momentum rank
        sorted_assets = sorted(momentum_ranks.items(), key=lambda x: x[1], reverse=True)

        # Select top and bottom performers
        num_positions = max(1, len(sorted_assets) // 4)  # Top/bottom quartile

        signals = {}
        for symbol in price_data_dict.keys():
            if symbol in momentum_ranks:
                rank = momentum_ranks[symbol]
                # Long top performers, short bottom performers
                if rank <= num_positions:  # Top performers
                    signals[symbol] = 1
                elif rank >= len(sorted_assets) - num_positions:  # Bottom performers
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0
            else:
                signals[symbol] = 0

        # Convert signals to DataFrames as expected by the algorithm
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class VolatilityRegimeStrategy(BaseStrategy):
    """Volatility regime switching strategy based on statistical process control"""

    def __init__(self, lookback_period=63, regime_threshold=1.5, min_regime_length=5):
        """
        Initialize volatility regime strategy

        Args:
            lookback_period (int): Period for calculating volatility statistics
            regime_threshold (float): Threshold for regime detection (standard deviations)
            min_regime_length (int): Minimum length for regime confirmation
        """
        super().__init__("Volatility Regime Strategy")
        self.lookback_period = lookback_period
        self.regime_threshold = regime_threshold
        self.min_regime_length = min_regime_length

    def detect_volatility_regime(self, price_data_dict, current_date):
        """
        Detect current volatility regime using statistical process control

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for regime detection

        Returns:
            str: Current regime ('low_vol', 'high_vol', 'normal')
        """
        symbols = list(price_data_dict.keys())
        volatility_scores = []

        for symbol in symbols:
            data = price_data_dict[symbol]

            if len(data) < self.lookback_period:
                continue

            # Calculate rolling volatility
            returns = data['Close'].pct_change()
            rolling_vol = returns.rolling(self.lookback_period).std()

            if current_date in rolling_vol.index:
                current_vol = rolling_vol.loc[current_date]

                if not pd.isna(current_vol):
                    # Calculate historical volatility statistics
                    historical_vol = rolling_vol.dropna()
                    if len(historical_vol) > 30:
                        vol_mean = historical_vol.mean()
                        vol_std = historical_vol.std()

                        # Z-score of current volatility
                        if vol_std > 0:
                            z_score = (current_vol - vol_mean) / vol_std
                            volatility_scores.append(z_score)

        if not volatility_scores:
            return 'normal'

        # Average volatility regime across assets
        avg_z_score = np.mean(volatility_scores)

        if avg_z_score > self.regime_threshold:
            return 'high_vol'
        elif avg_z_score < -self.regime_threshold:
            return 'low_vol'
        else:
            return 'normal'

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate trading signals based on volatility regime

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        # Detect current regime
        current_regime = self.detect_volatility_regime(price_data_dict, current_date)

        # Strategy logic based on regime
        signals = {}

        if current_regime == 'low_vol':
            # In low volatility, use mean reversion strategies
            for symbol in price_data_dict.keys():
                data = price_data_dict[symbol]

                if len(data) < 20:
                    signals[symbol] = 0
                    continue

                # Simple mean reversion signal
                current_price = data['Close'].loc[current_date]
                ma_20 = data['Close'].rolling(20).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_20):
                    price_deviation = (current_price - ma_20) / ma_20

                    if price_deviation < -0.05:  # 5% below mean
                        signals[symbol] = 1  # Long (expect reversion up)
                    elif price_deviation > 0.05:  # 5% above mean
                        signals[symbol] = -1  # Short (expect reversion down)
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        elif current_regime == 'high_vol':
            # In high volatility, use momentum/breakout strategies
            for symbol in price_data_dict.keys():
                data = price_data_dict[symbol]

                if len(data) < 20:
                    signals[symbol] = 0
                    continue

                # Volatility breakout signal
                current_price = data['Close'].loc[current_date]
                recent_high = data['High'].rolling(20).max().loc[current_date]
                recent_low = data['Low'].rolling(20).min().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(recent_high) and not pd.isna(recent_low):
                    breakout_up = current_price > recent_high * 0.995  # Near recent high
                    breakout_down = current_price < recent_low * 1.005  # Near recent low

                    if breakout_up:
                        signals[symbol] = 1  # Long breakout
                    elif breakout_down:
                        signals[symbol] = -1  # Short breakdown
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        else:  # Normal volatility
            # Use trend-following in normal volatility
            for symbol in price_data_dict.keys():
                data = price_data_dict[symbol]

                if len(data) < 50:
                    signals[symbol] = 0
                    continue

                # Moving average crossover signal
                current_price = data['Close'].loc[current_date]
                ma_10 = data['Close'].rolling(10).mean().loc[current_date]
                ma_30 = data['Close'].rolling(30).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_10) and not pd.isna(ma_30):
                    if current_price > ma_10 and ma_10 > ma_30:
                        signals[symbol] = 1  # Long trend
                    elif current_price < ma_10 and ma_10 < ma_30:
                        signals[symbol] = -1  # Short trend
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        # Convert signals to DataFrames as expected by the algorithm
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                # Create a DataFrame with the same index as the price data
                # Set signal to 0 for all dates except current_date where we set the actual signal
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class LiquidityTimingStrategy(BaseStrategy):
    """Liquidity timing strategy based on order flow and volume patterns"""

    def __init__(self, volume_lookback=20, price_impact_threshold=0.001,
                 liquidity_premium_threshold=0.02):
        """
        Initialize liquidity timing strategy

        Args:
            volume_lookback (int): Lookback period for volume analysis
            price_impact_threshold (float): Threshold for price impact detection
            liquidity_premium_threshold (float): Threshold for liquidity premium
        """
        super().__init__("Liquidity Timing Strategy")
        self.volume_lookback = volume_lookback
        self.price_impact_threshold = price_impact_threshold
        self.liquidity_premium_threshold = liquidity_premium_threshold

    def calculate_liquidity_metrics(self, data):
        """
        Calculate liquidity metrics for an asset

        Args:
            data (pd.DataFrame): Price and volume data

        Returns:
            pd.DataFrame: Liquidity metrics
        """
        df = data.copy()

        # Volume metrics
        df['volume_ma'] = df['Volume'].rolling(self.volume_lookback).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']

        # Price impact (volatility per unit volume)
        df['price_impact'] = df['Close'].pct_change().abs() / df['Volume']

        # Amihud illiquidity measure
        df['amihud'] = (df['Close'].pct_change().abs() * 1000000) / df['Volume']

        # Turnover ratio
        df['turnover'] = df['Volume'] / df['Volume'].rolling(252).mean()

        return df

    def detect_liquidity_events(self, data, current_date):
        """
        Detect significant liquidity events

        Args:
            data (pd.DataFrame): Price and volume data
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Liquidity event indicators
        """
        if current_date not in data.index:
            return {'event': 'none', 'strength': 0}

        # Get recent liquidity metrics
        recent_data = data.loc[max(data.index[0], current_date - pd.Timedelta(days=self.volume_lookback*2)):current_date]

        if len(recent_data) < self.volume_lookback:
            return {'event': 'none', 'strength': 0}

        # Calculate current metrics
        current_volume = data['Volume'].loc[current_date]
        current_price_impact = data['Close'].pct_change().loc[current_date]
        current_amihud = data['amihud'].loc[current_date] if 'amihud' in data.columns else 0

        # Detect volume spikes
        volume_threshold = recent_data['Volume'].quantile(0.8)
        volume_spike = current_volume > volume_threshold

        # Detect price impact events
        impact_threshold = recent_data['price_impact'].quantile(0.8)
        price_impact_event = abs(current_price_impact) > impact_threshold if not pd.isna(current_price_impact) else False

        # Detect liquidity drying up
        low_volume = current_volume < recent_data['Volume'].quantile(0.2)

        # Determine liquidity event
        if volume_spike and price_impact_event:
            return {'event': 'liquidity_shock', 'strength': 1.0, 'direction': 'down' if current_price_impact < 0 else 'up'}
        elif low_volume:
            return {'event': 'low_liquidity', 'strength': 0.8, 'direction': 'neutral'}
        elif price_impact_event:
            return {'event': 'price_impact', 'strength': 0.6, 'direction': 'down' if current_price_impact < 0 else 'up'}
        else:
            return {'event': 'normal', 'strength': 0.0, 'direction': 'neutral'}

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate trading signals based on liquidity timing

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        for symbol in price_data_dict.keys():
            data = price_data_dict[symbol]

            if len(data) < self.volume_lookback:
                signals[symbol] = 0
                continue

            # Calculate liquidity metrics
            liquidity_data = self.calculate_liquidity_metrics(data)

            # Detect liquidity events
            liquidity_event = self.detect_liquidity_events(liquidity_data, current_date)

            # Generate signals based on liquidity events
            if liquidity_event['event'] == 'liquidity_shock':
                # Fade the liquidity shock (contrarian approach)
                if liquidity_event['direction'] == 'down':
                    signals[symbol] = 1  # Long when price drops on volume spike
                else:
                    signals[symbol] = -1  # Short when price rises on volume spike
            elif liquidity_event['event'] == 'low_liquidity':
                # Avoid trading in low liquidity
                signals[symbol] = 0
            elif liquidity_event['event'] == 'price_impact':
                # Follow price impact direction
                if liquidity_event['direction'] == 'down':
                    signals[symbol] = -1  # Short on negative impact
                else:
                    signals[symbol] = 1   # Long on positive impact
            else:
                # Normal liquidity - use momentum
                current_price = data['Close'].loc[current_date]
                ma_10 = data['Close'].rolling(10).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_10):
                    if current_price > ma_10 * 1.02:  # 2% above moving average
                        signals[symbol] = 1
                    elif current_price < ma_10 * 0.98:  # 2% below moving average
                        signals[symbol] = -1
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        # Convert signals to DataFrames as expected by the algorithm
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                # Create a DataFrame with the same index as the price data
                # Set signal to 0 for all dates except current_date where we set the actual signal
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class StatisticalProcessControlStrategy(BaseStrategy):
    """Statistical process control strategy for detecting regime changes"""

    def __init__(self, control_limits=3.0, min_window=20, max_window=100):
        """
        Initialize statistical process control strategy

        Args:
            control_limits (float): Standard deviation limits for control charts
            min_window (int): Minimum window for statistical calculations
            max_window (int): Maximum window for statistical calculations
        """
        super().__init__("Statistical Process Control Strategy")
        self.control_limits = control_limits
        self.min_window = min_window
        self.max_window = max_window

    def calculate_control_limits(self, returns_series):
        """
        Calculate statistical process control limits

        Args:
            returns_series (pd.Series): Return series for analysis

        Returns:
            dict: Control limits and statistics
        """
        if len(returns_series) < self.min_window:
            return {
                'upper_limit': 0,
                'lower_limit': 0,
                'center_line': 0,
                'current_value': 0,
                'out_of_control': False
            }

        # Calculate rolling statistics
        rolling_mean = returns_series.rolling(self.min_window).mean()
        rolling_std = returns_series.rolling(self.min_window).std()

        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]

        if pd.isna(current_mean) or pd.isna(current_std) or current_std == 0:
            return {
                'upper_limit': 0,
                'lower_limit': 0,
                'center_line': 0,
                'current_value': 0,
                'out_of_control': False
            }

        # Control limits
        upper_limit = current_mean + self.control_limits * current_std
        lower_limit = current_mean - self.control_limits * current_std

        # Current value (latest return)
        current_value = returns_series.iloc[-1]

        # Check if out of control
        out_of_control = current_value > upper_limit or current_value < lower_limit

        return {
            'upper_limit': upper_limit,
            'lower_limit': lower_limit,
            'center_line': current_mean,
            'current_value': current_value,
            'out_of_control': out_of_control,
            'standard_deviation': current_std,
            'mean': current_mean
        }

    def detect_regime_changes(self, price_data_dict, current_date):
        """
        Detect regime changes using statistical process control

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for analysis

        Returns:
            dict: Regime change indicators
        """
        symbols = list(price_data_dict.keys())
        regime_indicators = {}

        for symbol in symbols:
            data = price_data_dict[symbol]

            if len(data) < self.min_window or current_date not in data.index:
                regime_indicators[symbol] = {'regime': 'insufficient_data', 'signal': 0}
                continue

            # Calculate returns
            returns = data['Close'].pct_change().dropna()

            # Calculate control limits
            control_stats = self.calculate_control_limits(returns)

            # Generate trading signal based on control chart
            if control_stats['out_of_control']:
                if control_stats['current_value'] > control_stats['upper_limit']:
                    # Above upper control limit - potential upward regime
                    regime_indicators[symbol] = {'regime': 'high_volatility', 'signal': 1}
                else:
                    # Below lower control limit - potential downward regime
                    regime_indicators[symbol] = {'regime': 'low_volatility', 'signal': -1}
            else:
                # Within control limits - normal regime
                regime_indicators[symbol] = {'regime': 'normal', 'signal': 0}

        return regime_indicators

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate trading signals based on statistical process control

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        # Detect regime changes
        regime_indicators = self.detect_regime_changes(price_data_dict, current_date)

        # Generate signals based on regime analysis
        signals = {}

        for symbol, indicator in regime_indicators.items():
            if indicator['regime'] == 'high_volatility':
                # In high volatility regime, use breakout strategies
                data = price_data_dict[symbol]

                if len(data) < 20:
                    signals[symbol] = 0
                    continue

                current_price = data['Close'].loc[current_date]
                recent_high = data['High'].rolling(20).max().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(recent_high):
                    if current_price > recent_high * 0.99:  # Near recent high
                        signals[symbol] = 1  # Long breakout
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

            elif indicator['regime'] == 'low_volatility':
                # In low volatility regime, use mean reversion
                data = price_data_dict[symbol]

                if len(data) < 20:
                    signals[symbol] = 0
                    continue

                current_price = data['Close'].loc[current_date]
                ma_20 = data['Close'].rolling(20).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_20):
                    deviation = (current_price - ma_20) / ma_20

                    if deviation < -0.03:  # 3% below mean
                        signals[symbol] = 1  # Long mean reversion
                    elif deviation > 0.03:  # 3% above mean
                        signals[symbol] = -1  # Short mean reversion
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

            else:  # Normal regime
                # Use trend following in normal regime
                data = price_data_dict[symbol]

                if len(data) < 50:
                    signals[symbol] = 0
                    continue

                current_price = data['Close'].loc[current_date]
                ma_10 = data['Close'].rolling(10).mean().loc[current_date]
                ma_30 = data['Close'].rolling(30).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_10) and not pd.isna(ma_30):
                    if current_price > ma_10 and ma_10 > ma_30:
                        signals[symbol] = 1  # Long trend
                    elif current_price < ma_10 and ma_10 < ma_30:
                        signals[symbol] = -1  # Short trend
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        # Convert signals to DataFrames as expected by the algorithm
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                # Create a DataFrame with the same index as the price data
                # Set signal to 0 for all dates except current_date where we set the actual signal
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes

