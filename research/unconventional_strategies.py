"""
Unconventional Quantitative Strategies for Alpha Generation

This module implements highly unconventional quantitative strategies that go beyond
traditional factor models and statistical arbitrage. These strategies draw from
behavioral finance, information theory, complex systems, and other non-traditional
approaches to quantitative finance.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from trading.strategies import BaseStrategy


class AttentionDrivenStrategy(BaseStrategy):
    """Behavioral finance strategy exploiting investor attention patterns"""

    def __init__(self, attention_lookback=21, attention_threshold=1.5,
                 volume_multiplier=2.0, price_impact_window=5):
        """
        Initialize attention-driven strategy

        Args:
            attention_lookback (int): Period for calculating attention metrics
            attention_threshold (float): Threshold for attention spikes
            volume_multiplier (float): Volume multiplier for attention signals
            price_impact_window (int): Window for measuring price impact
        """
        super().__init__("Attention-Driven Strategy")
        self.attention_lookback = attention_lookback
        self.attention_threshold = attention_threshold
        self.volume_multiplier = volume_multiplier
        self.price_impact_window = price_impact_window

    def calculate_attention_metrics(self, data):
        """
        Calculate investor attention metrics from price and volume data

        Args:
            data (pd.DataFrame): Price and volume data

        Returns:
            pd.DataFrame: Attention metrics
        """
        df = data.copy()

        # Trading volume as proxy for attention
        df['volume_ma'] = df['Volume'].rolling(self.attention_lookback).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']

        # Price volatility as attention proxy
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.attention_lookback).std()

        # Abnormal trading activity (attention spikes)
        df['attention_spike'] = (df['volume_ratio'] > self.attention_threshold).astype(int)

        # Price impact of attention
        df['price_impact'] = df['returns'].abs() * df['volume_ratio']

        # Cumulative attention over time
        df['attention_accumulation'] = df['attention_spike'].rolling(self.attention_lookback).sum()

        return df

    def detect_attention_regimes(self, data, current_date):
        """
        Detect attention-driven market regimes

        Args:
            data (pd.DataFrame): Enhanced data with attention metrics
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Attention regime indicators
        """
        if current_date not in data.index:
            return {'regime': 'insufficient_data', 'attention_level': 0}

        recent_data = data.loc[:current_date].tail(self.attention_lookback)

        # Calculate attention metrics
        avg_attention = recent_data['attention_accumulation'].mean()
        current_spike = recent_data['attention_spike'].iloc[-1]
        price_impact = recent_data['price_impact'].tail(self.price_impact_window).mean()

        # Determine attention regime
        if avg_attention > self.attention_lookback * 0.3:  # High attention period
            if current_spike == 1 and price_impact > recent_data['price_impact'].quantile(0.8):
                regime = 'attention_shock'
            else:
                regime = 'high_attention'
        elif avg_attention < self.attention_lookback * 0.1:  # Low attention period
            regime = 'low_attention'
        else:
            regime = 'normal_attention'

        return {
            'regime': regime,
            'attention_level': avg_attention / self.attention_lookback,
            'current_spike': current_spike,
            'price_impact': price_impact
        }

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate trading signals based on attention patterns

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        for symbol in price_data_dict.keys():
            data = price_data_dict[symbol]

            if len(data) < self.attention_lookback:
                signals[symbol] = 0
                continue

            # Calculate attention metrics
            attention_data = self.calculate_attention_metrics(data)

            # Detect attention regime
            attention_regime = self.detect_attention_regimes(attention_data, current_date)

            # Generate signals based on attention regime
            if attention_regime['regime'] == 'attention_shock':
                # During attention shocks, fade the move (contrarian)
                current_return = attention_data['returns'].loc[current_date]
                if not pd.isna(current_return):
                    signals[symbol] = -1 if current_return > 0 else 1
                else:
                    signals[symbol] = 0

            elif attention_regime['regime'] == 'high_attention':
                # High attention periods - use momentum
                recent_returns = attention_data['returns'].tail(5).mean()
                if recent_returns > 0.005:  # 0.5% daily return
                    signals[symbol] = 1
                elif recent_returns < -0.005:
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0

            elif attention_regime['regime'] == 'low_attention':
                # Low attention periods - mean reversion
                current_price = data['Close'].loc[current_date]
                ma_20 = data['Close'].rolling(20).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_20):
                    deviation = (current_price - ma_20) / ma_20
                    if deviation < -0.02:  # 2% below mean
                        signals[symbol] = 1
                    elif deviation > 0.02:  # 2% above mean
                        signals[symbol] = -1
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

            else:  # Normal attention
                # Trend following in normal periods
                ma_10 = data['Close'].rolling(10).mean().loc[current_date]
                ma_30 = data['Close'].rolling(30).mean().loc[current_date]

                if not pd.isna(ma_10) and not pd.isna(ma_30):
                    if ma_10 > ma_30:
                        signals[symbol] = 1
                    elif ma_10 < ma_30:
                        signals[symbol] = -1
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        # Convert signals to DataFrames
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class SentimentRegimeStrategy(BaseStrategy):
    """Strategy exploiting market sentiment regimes and behavioral biases"""

    def __init__(self, sentiment_lookback=63, extreme_sentiment_threshold=2.0,
                 herding_lookback=21, anchoring_window=252):
        """
        Initialize sentiment regime strategy

        Args:
            sentiment_lookback (int): Period for sentiment calculation
            extreme_sentiment_threshold (float): Threshold for extreme sentiment
            herding_lookback (int): Lookback for herding behavior detection
            anchoring_window (int): Window for anchoring bias detection
        """
        super().__init__("Sentiment Regime Strategy")
        self.sentiment_lookback = sentiment_lookback
        self.extreme_sentiment_threshold = extreme_sentiment_threshold
        self.herding_lookback = herding_lookback
        self.anchoring_window = anchoring_window

    def calculate_sentiment_indicators(self, data):
        """
        Calculate behavioral sentiment indicators

        Args:
            data (pd.DataFrame): Price and volume data

        Returns:
            pd.DataFrame: Sentiment indicators
        """
        df = data.copy()

        # Returns for sentiment analysis
        df['returns'] = df['Close'].pct_change()

        # Market sentiment proxy (extreme returns as fear/greed indicator)
        df['extreme_returns'] = (df['returns'].abs() > df['returns'].rolling(self.sentiment_lookback).std() * 2).astype(int)

        # Herding behavior (correlation with market)
        # Using volume-weighted returns as market proxy
        df['vw_returns'] = df['returns'] * df['Volume']
        df['market_correlation'] = df['vw_returns'].rolling(self.herding_lookback).corr(df['returns'])

        # Anchoring bias (deviation from long-term moving averages)
        df['ma_long'] = df['Close'].rolling(self.anchoring_window).mean()
        df['anchoring_deviation'] = (df['Close'] - df['ma_long']) / df['ma_long']

        # Momentum as sentiment proxy
        df['momentum'] = df['returns'].rolling(self.sentiment_lookback).mean()

        return df

    def detect_sentiment_regime(self, sentiment_data, current_date):
        """
        Detect current sentiment regime

        Args:
            sentiment_data (pd.DataFrame): Data with sentiment indicators
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Sentiment regime analysis
        """
        if current_date not in sentiment_data.index:
            return {'regime': 'insufficient_data', 'sentiment_score': 0}

        recent_data = sentiment_data.loc[:current_date].tail(self.sentiment_lookback)

        # Calculate composite sentiment score
        extreme_freq = recent_data['extreme_returns'].mean()
        herding_intensity = recent_data['market_correlation'].mean()
        anchoring_bias = abs(recent_data['anchoring_deviation'].iloc[-1])
        momentum_sentiment = recent_data['momentum'].iloc[-1]

        # Composite sentiment score (-1 to 1, negative = fear, positive = greed)
        sentiment_score = (
            (extreme_freq - 0.1) * 2 +  # Extreme returns frequency
            herding_intensity +  # Herding behavior
            (anchoring_bias - 0.1) * 5 +  # Anchoring deviation
            momentum_sentiment * 10  # Momentum sentiment
        ) / 4

        # Determine regime
        if sentiment_score > self.extreme_sentiment_threshold:
            regime = 'extreme_greed'
        elif sentiment_score < -self.extreme_sentiment_threshold:
            regime = 'extreme_fear'
        elif abs(sentiment_score) > 1.0:
            regime = 'elevated_sentiment'
        else:
            regime = 'neutral_sentiment'

        return {
            'regime': regime,
            'sentiment_score': sentiment_score,
            'extreme_frequency': extreme_freq,
            'herding_intensity': herding_intensity,
            'anchoring_bias': anchoring_bias
        }

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate signals based on sentiment regime

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        for symbol in price_data_dict.keys():
            data = price_data_dict[symbol]

            if len(data) < self.sentiment_lookback:
                signals[symbol] = 0
                continue

            # Calculate sentiment indicators
            sentiment_data = self.calculate_sentiment_indicators(data)

            # Detect sentiment regime
            sentiment_regime = self.detect_sentiment_regime(sentiment_data, current_date)

            # Generate signals based on regime
            if sentiment_regime['regime'] == 'extreme_greed':
                # In extreme greed, fade momentum (contrarian)
                recent_momentum = sentiment_data['momentum'].loc[current_date]
                if not pd.isna(recent_momentum):
                    signals[symbol] = -1 if recent_momentum > 0 else 1
                else:
                    signals[symbol] = 0

            elif sentiment_regime['regime'] == 'extreme_fear':
                # In extreme fear, buy the dip (momentum)
                recent_momentum = sentiment_data['momentum'].loc[current_date]
                if not pd.isna(recent_momentum):
                    signals[symbol] = 1 if recent_momentum < 0 else -1
                else:
                    signals[symbol] = 0

            elif sentiment_regime['regime'] == 'elevated_sentiment':
                # High sentiment - use anchoring bias
                anchoring_dev = sentiment_data['anchoring_deviation'].loc[current_date]
                if not pd.isna(anchoring_dev):
                    if anchoring_dev > 0.05:  # Over-anchored to high prices
                        signals[symbol] = -1  # Short reversion
                    elif anchoring_dev < -0.05:  # Under-anchored to low prices
                        signals[symbol] = 1   # Long reversion
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

            else:  # Neutral sentiment
                # Use herding behavior for signals
                herding_corr = sentiment_data['market_correlation'].loc[current_date]
                if not pd.isna(herding_corr):
                    if herding_corr > 0.8:  # Strong herding
                        signals[symbol] = 1  # Follow the herd
                    elif herding_corr < 0.2:  # Weak herding
                        signals[symbol] = -1  # Contrarian
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0

        # Convert signals to DataFrames
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class InformationTheoryStrategy(BaseStrategy):
    """Strategy using information theory concepts for signal generation"""

    def __init__(self, entropy_window=100, transfer_entropy_lags=5,
                 mutual_info_threshold=0.1, complexity_lookback=50):
        """
        Initialize information theory strategy

        Args:
            entropy_window (int): Window for entropy calculations
            transfer_entropy_lags (int): Maximum lags for transfer entropy
            mutual_info_threshold (float): Threshold for mutual information signals
            complexity_lookback (int): Lookback for complexity measures
        """
        super().__init__("Information Theory Strategy")
        self.entropy_window = entropy_window
        self.transfer_entropy_lags = transfer_entropy_lags
        self.mutual_info_threshold = mutual_info_threshold
        self.complexity_lookback = complexity_lookback

    def calculate_entropy_measures(self, returns_series):
        """
        Calculate various entropy measures for the return series

        Args:
            returns_series (pd.Series): Return series

        Returns:
            pd.Series: Entropy measures
        """
        # Approximate entropy (simplified)
        def approximate_entropy(series, m=2, r=0.2):
            """Simplified approximate entropy calculation"""
            if len(series) < m + 1:
                return 0

            # Create embedded vectors
            embedded = []
            for i in range(len(series) - m + 1):
                embedded.append(series.iloc[i:i+m].values)

            if len(embedded) < 2:
                return 0

            # Calculate distances and patterns
            distances = []
            for i in range(len(embedded)):
                for j in range(i + 1, len(embedded)):
                    dist = np.max(np.abs(embedded[i] - embedded[j]))
                    distances.append(dist)

            if not distances:
                return 0

            # Count similar patterns
            r_threshold = r * np.std(series)
            similar_patterns = sum(1 for d in distances if d <= r_threshold)

            if similar_patterns == 0:
                return 0

            # Approximate entropy
            phi_m = similar_patterns / len(distances)
            return -np.log(phi_m) if phi_m > 0 else 0

        # Rolling entropy calculation
        entropy_series = returns_series.rolling(self.entropy_window).apply(approximate_entropy)

        return entropy_series

    def calculate_transfer_entropy(self, source_series, target_series, max_lag=5):
        """
        Calculate transfer entropy from source to target

        Args:
            source_series (pd.Series): Source time series
            target_series (pd.Series): Target time series
            max_lag (int): Maximum lag to consider

        Returns:
            dict: Transfer entropy measures
        """
        if len(source_series) < max_lag + 10 or len(target_series) < max_lag + 10:
            return {'te_value': 0, 'optimal_lag': 0, 'directionality': 0}

        te_values = []

        for lag in range(1, max_lag + 1):
            # Create lagged source
            lagged_source = source_series.shift(lag)

            # Remove NaN values
            combined = pd.DataFrame({
                'target': target_series,
                'lagged_source': lagged_source
            }).dropna()

            if len(combined) < 20:
                te_values.append(0)
                continue

            # Calculate conditional mutual information
            # Simplified: correlation between lagged source and target
            te_value = abs(combined['target'].corr(combined['lagged_source']))
            te_values.append(te_value)

        # Find optimal lag
        optimal_lag = np.argmax(te_values) + 1 if te_values else 0
        max_te = max(te_values) if te_values else 0

        return {
            'te_value': max_te,
            'optimal_lag': optimal_lag,
            'directionality': max_te
        }

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate signals using information theory concepts

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        # Get all symbols for cross-asset analysis
        symbols = list(price_data_dict.keys())

        for symbol in symbols:
            data = price_data_dict[symbol]

            if len(data) < self.entropy_window:
                signals[symbol] = 0
                continue

            # Calculate returns
            returns = data['Close'].pct_change().dropna()

            if len(returns) < self.entropy_window:
                signals[symbol] = 0
                continue

            # Calculate entropy measures
            entropy_measures = self.calculate_entropy_measures(returns)

            # Get current entropy
            if current_date in entropy_measures.index:
                entropy_slice = entropy_measures.loc[:current_date]
                if isinstance(entropy_slice, pd.Series) and len(entropy_slice) > 0:
                    current_entropy = entropy_slice.iloc[-1]
                else:
                    current_entropy = entropy_slice if pd.notna(entropy_slice) else 0
            else:
                current_entropy = 0

            # Calculate transfer entropy with other assets
            te_signals = []

            for other_symbol in symbols:
                if other_symbol != symbol:
                    other_data = price_data_dict[other_symbol]
                    other_returns = other_data['Close'].pct_change().dropna()

                    # Calculate transfer entropy
                    te_analysis = self.calculate_transfer_entropy(other_returns, returns, self.transfer_entropy_lags)
                    te_signals.append(te_analysis['te_value'])

            # Average transfer entropy signal
            avg_te_signal = np.mean(te_signals) if te_signals else 0

            # Generate composite signal
            if len(entropy_measures.dropna()) >= 10:
                entropy_80th = entropy_measures.quantile(0.8)
                entropy_20th = entropy_measures.quantile(0.2)
                entropy_signal = 1 if current_entropy > entropy_80th else -1 if current_entropy < entropy_20th else 0
            else:
                entropy_signal = 0
            te_signal = 1 if avg_te_signal > self.mutual_info_threshold else 0

            # Combine signals
            combined_signal = entropy_signal + te_signal

            if combined_signal > 1:
                signals[symbol] = 1
            elif combined_signal < -1:
                signals[symbol] = -1
            else:
                signals[symbol] = 0

        # Convert signals to DataFrames
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class ComplexSystemsStrategy(BaseStrategy):
    """Strategy using complex systems theory and network effects"""

    def __init__(self, network_lookback=100, centrality_threshold=0.7,
                 contagion_window=21, synchronization_lags=10):
        """
        Initialize complex systems strategy

        Args:
            network_lookback (int): Lookback for network analysis
            centrality_threshold (float): Threshold for centrality signals
            contagion_window (int): Window for contagion detection
            synchronization_lags (int): Maximum lags for synchronization analysis
        """
        super().__init__("Complex Systems Strategy")
        self.network_lookback = network_lookback
        self.centrality_threshold = centrality_threshold
        self.contagion_window = contagion_window
        self.synchronization_lags = synchronization_lags

    def calculate_network_centrality(self, price_data_dict, current_date):
        """
        Calculate network centrality measures for assets

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Centrality scores for each asset
        """
        symbols = list(price_data_dict.keys())
        centrality_scores = {}

        # Calculate correlation matrix
        correlations = {}
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                data1 = price_data_dict[symbol1]
                data2 = price_data_dict[symbol2]

                if current_date in data1.index and current_date in data2.index:
                    returns1 = data1['Close'].pct_change().loc[:current_date].tail(self.network_lookback)
                    returns2 = data2['Close'].pct_change().loc[:current_date].tail(self.network_lookback)

                    if len(returns1) > 30 and len(returns2) > 30:
                        corr = abs(returns1.corr(returns2))
                        correlations[(symbol1, symbol2)] = corr
                        correlations[(symbol2, symbol1)] = corr

        # Calculate degree centrality (number of strong connections)
        for symbol in symbols:
            strong_connections = 0
            for other_symbol in symbols:
                if other_symbol != symbol:
                    key = (symbol, other_symbol)
                    if key in correlations and correlations[key] > 0.5:
                        strong_connections += 1

            centrality_scores[symbol] = strong_connections / max(1, len(symbols) - 1)

        return centrality_scores

    def detect_contagion_effects(self, price_data_dict, current_date):
        """
        Detect contagion effects across the asset network

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Contagion indicators
        """
        symbols = list(price_data_dict.keys())
        contagion_indicators = {}

        for symbol in symbols:
            data = price_data_dict[symbol]

            if current_date not in data.index:
                contagion_indicators[symbol] = {'contagion_risk': 0, 'volatility_contagion': 0}
                continue

            # Calculate extreme co-movements
            returns = data['Close'].pct_change().loc[:current_date].tail(self.contagion_window)
            extreme_returns = (returns.abs() > returns.std() * 2).sum()
            contagion_risk = extreme_returns / self.contagion_window

            # Calculate volatility clustering
            volatility = returns.rolling(5).std()
            vol_autocorr = volatility.autocorr(lag=1) if len(volatility.dropna()) > 10 else 0

            contagion_indicators[symbol] = {
                'contagion_risk': contagion_risk,
                'volatility_contagion': vol_autocorr
            }

        return contagion_indicators

    def analyze_synchronization(self, price_data_dict, current_date):
        """
        Analyze synchronization patterns across assets

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Synchronization analysis
        """
        symbols = list(price_data_dict.keys())
        sync_analysis = {}

        for symbol in symbols:
            data = price_data_dict[symbol]

            if current_date not in data.index:
                sync_analysis[symbol] = {'sync_strength': 0, 'phase_locking': 0}
                continue

            returns = data['Close'].pct_change().loc[:current_date].tail(self.synchronization_lags * 2)

            if len(returns) < self.synchronization_lags:
                sync_analysis[symbol] = {'sync_strength': 0, 'phase_locking': 0}
                continue

            # Calculate synchronization with market (average of all assets)
            market_returns = []
            for other_symbol in symbols:
                other_data = price_data_dict[other_symbol]
                if current_date in other_data.index:
                    other_returns = other_data['Close'].pct_change().loc[:current_date].tail(self.synchronization_lags * 2)
                    market_returns.append(other_returns)

            if market_returns:
                avg_market_returns = pd.concat(market_returns, axis=1).mean(axis=1)

                # Cross-correlation for synchronization
                sync_corr = abs(returns.corr(avg_market_returns)) if len(avg_market_returns) == len(returns) else 0

                # Phase locking (simplified)
                phase_locking = sync_corr ** 2

                sync_analysis[symbol] = {
                    'sync_strength': sync_corr,
                    'phase_locking': phase_locking
                }
            else:
                sync_analysis[symbol] = {'sync_strength': 0, 'phase_locking': 0}

        return sync_analysis

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate signals using complex systems theory

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        # Calculate network metrics
        centrality = self.calculate_network_centrality(price_data_dict, current_date)
        contagion = self.detect_contagion_effects(price_data_dict, current_date)
        synchronization = self.analyze_synchronization(price_data_dict, current_date)

        for symbol in price_data_dict.keys():
            data = price_data_dict[symbol]

            if current_date not in data.index:
                signals[symbol] = 0
                continue

            # Combine complex systems signals
            cent_score = centrality.get(symbol, 0)
            cont_score = contagion.get(symbol, {}).get('contagion_risk', 0)
            sync_score = synchronization.get(symbol, {}).get('sync_strength', 0)

            # Generate composite signal
            # High centrality + low contagion = potential for trending moves
            # High contagion + high synchronization = risk of contagion effects
            network_signal = cent_score - cont_score - sync_score

            # Convert to trading signal
            if network_signal > self.centrality_threshold:
                signals[symbol] = 1  # High centrality, low contagion - bullish
            elif network_signal < -self.centrality_threshold:
                signals[symbol] = -1  # High contagion, high synchronization - bearish
            else:
                signals[symbol] = 0

        # Convert signals to DataFrames
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class FractalChaosStrategy(BaseStrategy):
    """Strategy using fractal geometry and chaos theory concepts"""

    def __init__(self, fractal_window=200, hurst_lookback=100,
                 lyapunov_window=50, chaos_threshold=0.1):
        """
        Initialize fractal chaos strategy

        Args:
            fractal_window (int): Window for fractal dimension calculations
            hurst_lookback (int): Lookback for Hurst exponent
            lyapunov_window (int): Window for Lyapunov exponent
            chaos_threshold (float): Threshold for chaos detection
        """
        super().__init__("Fractal Chaos Strategy")
        self.fractal_window = fractal_window
        self.hurst_lookback = hurst_lookback
        self.lyapunov_window = lyapunov_window
        self.chaos_threshold = chaos_threshold

    def calculate_hurst_exponent(self, price_series):
        """
        Calculate Hurst exponent for fractal analysis

        Args:
            price_series (pd.Series): Price series

        Returns:
            float: Hurst exponent
        """
        if len(price_series) < 20:
            return 0.5  # Random walk

        # R/S analysis for Hurst exponent
        def rs_analysis(series):
            n = len(series)
            mean = series.mean()
            cumulative = (series - mean).cumsum()
            r = cumulative.max() - cumulative.min()
            s = series.std()

            if s == 0:
                return 0

            return r / s

        # Calculate for different window sizes
        window_sizes = [4, 8, 16, 32, 64, 128]
        rs_values = []
        n_values = []

        for window in window_sizes:
            if len(price_series) >= window:
                rs = rs_analysis(price_series.tail(window))
                rs_values.append(rs)
                n_values.append(window)

        if len(rs_values) < 3:
            return 0.5

        # Linear regression on log-log plot
        log_n = np.log(n_values)
        log_rs = np.log(rs_values)

        slope, intercept = np.polyfit(log_n, log_rs, 1)
        hurst = slope

        return max(0, min(1, hurst))  # Constrain to [0,1]

    def calculate_fractal_dimension(self, price_series):
        """
        Calculate fractal dimension using box-counting method

        Args:
            price_series (pd.Series): Price series

        Returns:
            float: Fractal dimension
        """
        if len(price_series) < 20:
            return 1.5  # Brownian motion dimension

        # Simplified box-counting for time series
        returns = price_series.pct_change().dropna()

        # Calculate volatility scaling
        scales = [2, 4, 8, 16, 32]
        fluctuations = []

        for scale in scales:
            if len(returns) >= scale:
                # Average fluctuation over scale
                scaled_returns = returns.rolling(scale).std().dropna()
                fluctuation = scaled_returns.mean()
                fluctuations.append(fluctuation)

        if len(fluctuations) < 3:
            return 1.5

        # Fit power law: fluctuation ~ scale^(H-1)
        # Fractal dimension D = 2 - H for time series
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluctuations = np.log(fluctuations)

        slope, intercept = np.polyfit(log_scales, log_fluctuations, 1)
        hurst = slope + 1  # Since fluctuation ~ scale^(H-1), so H-1 = slope, H = slope + 1

        fractal_dimension = 2 - hurst

        return max(1, min(2, fractal_dimension))

    def calculate_lyapunov_exponent(self, price_series):
        """
        Calculate Lyapunov exponent for chaos detection

        Args:
            price_series (pd.Series): Price series

        Returns:
            float: Lyapunov exponent
        """
        if len(price_series) < self.lyapunov_window:
            return 0

        returns = price_series.pct_change().dropna()

        if len(returns) < 20:
            return 0

        # Simplified Lyapunov exponent using autocorrelation of absolute returns
        # High autocorrelation suggests deterministic (possibly chaotic) behavior
        abs_returns = returns.abs()

        try:
            autocorr_1 = abs_returns.autocorr(lag=1)
            autocorr_2 = abs_returns.autocorr(lag=2)

            # Lyapunov-like measure: rate of divergence in return patterns
            if autocorr_1 > 0:
                lyapunov_proxy = -np.log(abs(autocorr_1)) if abs(autocorr_1) > 0.1 else 0
            else:
                lyapunov_proxy = 0

            return lyapunov_proxy
        except:
            return 0

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate signals using fractal and chaos theory

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        for symbol in price_data_dict.keys():
            data = price_data_dict[symbol]

            if current_date not in data.index or len(data) < self.fractal_window:
                signals[symbol] = 0
                continue

            # Get recent price series
            recent_prices = data['Close'].loc[:current_date].tail(self.fractal_window)

            # Calculate fractal metrics
            hurst = self.calculate_hurst_exponent(recent_prices)
            fractal_dim = self.calculate_fractal_dimension(recent_prices)
            lyapunov = self.calculate_lyapunov_exponent(recent_prices)

            # Generate signals based on fractal properties
            # Hurst > 0.5: persistent (trending) behavior
            # Hurst < 0.5: anti-persistent (mean-reverting) behavior
            # High fractal dimension: complex behavior
            # Positive Lyapunov: chaotic behavior

            hurst_signal = 1 if hurst > 0.6 else -1 if hurst < 0.4 else 0
            chaos_signal = 1 if abs(lyapunov) > self.chaos_threshold else 0
            fractal_signal = 1 if fractal_dim > 1.7 else -1 if fractal_dim < 1.3 else 0

            # Combine signals
            combined_signal = hurst_signal + chaos_signal + fractal_signal

            if combined_signal >= 2:
                signals[symbol] = 1  # Strong trending/chaotic regime
            elif combined_signal <= -2:
                signals[symbol] = -1  # Strong mean-reverting/simple regime
            else:
                signals[symbol] = 0

        # Convert signals to DataFrames
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes


class QuantumInspiredStrategy(BaseStrategy):
    """Strategy inspired by quantum mechanics concepts"""

    def __init__(self, superposition_window=50, entanglement_threshold=0.8,
                 wave_function_lookback=30, quantum_state_periods=5):
        """
        Initialize quantum-inspired strategy

        Args:
            superposition_window (int): Window for superposition analysis
            entanglement_threshold (float): Threshold for quantum entanglement detection
            wave_function_lookback (int): Lookback for wave function analysis
            quantum_state_periods (int): Number of quantum state periods to analyze
        """
        super().__init__("Quantum-Inspired Strategy")
        self.superposition_window = superposition_window
        self.entanglement_threshold = entanglement_threshold
        self.wave_function_lookback = wave_function_lookback
        self.quantum_state_periods = quantum_state_periods

    def calculate_quantum_states(self, returns_series):
        """
        Calculate 'quantum states' from return patterns

        Args:
            returns_series (pd.Series): Return series

        Returns:
            dict: Quantum state analysis
        """
        if len(returns_series) < 20:
            return {'superposition': 0, 'entanglement': 0, 'wave_probability': 0.5}

        # 'Superposition' - measure of mixed positive/negative returns
        positive_returns = (returns_series > 0).sum()
        total_returns = len(returns_series)
        superposition = min(positive_returns, total_returns - positive_returns) / (total_returns / 2)

        # 'Wave function' - probability distribution of returns
        wave_probability = abs(returns_series.mean())  # Distance from zero

        # 'Entanglement' - correlation with its own lagged version
        autocorr = abs(returns_series.autocorr(lag=1)) if len(returns_series) > 10 else 0

        return {
            'superposition': superposition,
            'entanglement': autocorr,
            'wave_probability': wave_probability
        }

    def detect_quantum_regime(self, price_data_dict, current_date):
        """
        Detect quantum-like market regimes

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Quantum regime analysis
        """
        symbols = list(price_data_dict.keys())
        quantum_regimes = {}

        for symbol in symbols:
            data = price_data_dict[symbol]

            if current_date not in data.index:
                quantum_regimes[symbol] = {'regime': 'classical', 'coherence': 0}
                continue

            returns = data['Close'].pct_change().loc[:current_date].tail(self.superposition_window)

            if len(returns) < 10:
                quantum_regimes[symbol] = {'regime': 'classical', 'coherence': 0}
                continue

            quantum_states = self.calculate_quantum_states(returns)

            # Determine quantum regime
            coherence = (
                quantum_states['superposition'] +
                quantum_states['entanglement'] +
                quantum_states['wave_probability']
            ) / 3

            if coherence > 0.7:
                regime = 'quantum_coherent'  # Strong patterns, trending
            elif coherence < 0.3:
                regime = 'quantum_decoherent'  # Random, mean-reverting
            else:
                regime = 'quantum_superposition'  # Mixed signals

            quantum_regimes[symbol] = {
                'regime': regime,
                'coherence': coherence,
                'superposition': quantum_states['superposition']
            }

        return quantum_regimes

    def generate_signals(self, price_data_dict, current_date):
        """
        Generate signals using quantum-inspired concepts

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            current_date (pd.Timestamp): Current date for signal generation

        Returns:
            dict: Trading signals for each asset
        """
        signals = {}

        # Detect quantum regimes
        quantum_regimes = self.detect_quantum_regime(price_data_dict, current_date)

        for symbol in price_data_dict.keys():
            data = price_data_dict[symbol]

            if current_date not in data.index:
                signals[symbol] = 0
                continue

            regime_info = quantum_regimes.get(symbol, {'regime': 'classical', 'coherence': 0})

            # Generate signals based on quantum regime
            if regime_info['regime'] == 'quantum_coherent':
                # High coherence - follow the trend
                ma_short = data['Close'].rolling(10).mean().loc[current_date]
                ma_long = data['Close'].rolling(30).mean().loc[current_date]

                if not pd.isna(ma_short) and not pd.isna(ma_long):
                    signals[symbol] = 1 if ma_short > ma_long else -1
                else:
                    signals[symbol] = 0

            elif regime_info['regime'] == 'quantum_decoherent':
                # Low coherence - mean reversion
                current_price = data['Close'].loc[current_date]
                ma_20 = data['Close'].rolling(20).mean().loc[current_date]

                if not pd.isna(current_price) and not pd.isna(ma_20):
                    deviation = (current_price - ma_20) / ma_20
                    signals[symbol] = -1 if deviation > 0.02 else 1 if deviation < -0.02 else 0
                else:
                    signals[symbol] = 0

            elif regime_info['regime'] == 'quantum_superposition':
                # Mixed state - use momentum with caution
                momentum = data['Close'].pct_change().rolling(5).mean().loc[current_date]

                if not pd.isna(momentum):
                    signals[symbol] = 1 if momentum > 0.005 else -1 if momentum < -0.005 else 0
                else:
                    signals[symbol] = 0

            else:  # Classical regime
                # Traditional moving average crossover
                ma_5 = data['Close'].rolling(5).mean().loc[current_date]
                ma_20 = data['Close'].rolling(20).mean().loc[current_date]

                if not pd.isna(ma_5) and not pd.isna(ma_20):
                    signals[symbol] = 1 if ma_5 > ma_20 else -1
                else:
                    signals[symbol] = 0

        # Convert signals to DataFrames
        signal_dataframes = {}
        for symbol, signal in signals.items():
            if symbol in price_data_dict:
                data = price_data_dict[symbol]
                signal_df = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                signal_df.loc[current_date, 'signal'] = signal
                signal_dataframes[symbol] = signal_df

        return signal_dataframes
