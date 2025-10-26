"""
Simple trading strategies for algorithmic trading
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller


class BaseStrategy:
    """Base class for trading strategies"""

    def __init__(self, name):
        """Initialize strategy with name"""
        self.name = name

    def generate_signals(self, data):
        """
        Generate trading signals from market data
        Returns: DataFrame with 'signal' column (-1, 0, 1)
        """
        raise NotImplementedError("Subclasses must implement generate_signals")


class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover Strategy"""

    def __init__(self, short_window=5, long_window=20):
        """
        Initialize MA Crossover strategy

        Args:
            short_window (int): Short moving average period
            long_window (int): Long moving average period
        """
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """
        Generate buy/sell signals based on moving average crossover

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate moving averages
        df['ma_short'] = df['Close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['Close'].rolling(window=self.long_window).mean()

        # Generate signals
        df['signal'] = 0

        # Buy signal: short MA above long MA (trend up)
        buy_condition = df['ma_short'] > df['ma_long']
        df.loc[buy_condition, 'signal'] = 1

        # Sell signal: short MA below long MA (trend down)
        sell_condition = df['ma_short'] < df['ma_long']
        df.loc[sell_condition, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['ma_short', 'ma_long'], axis=1)

        return df


class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) Strategy"""

    def __init__(self, period=14, overbought=65, oversold=35):
        """
        Initialize RSI strategy

        Args:
            period (int): RSI calculation period
            overbought (float): Overbought threshold
            oversold (float): Oversold threshold
        """
        super().__init__("RSI Strategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        delta = prices.diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data):
        """
        Generate buy/sell signals based on RSI levels

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['Close'])

        # Generate signals
        df['signal'] = 0
        df['prev_rsi'] = df['rsi'].shift(1)

        # Buy signal: RSI is oversold or crosses above oversold level
        buy_condition = (df['rsi'] <= self.oversold) | ((df['prev_rsi'] <= self.oversold) & (df['rsi'] > self.oversold))
        df.loc[buy_condition, 'signal'] = 1

        # Sell signal: RSI is overbought or crosses below overbought level
        sell_condition = (df['rsi'] >= self.overbought) | ((df['prev_rsi'] >= self.overbought) & (df['rsi'] < self.overbought))
        df.loc[sell_condition, 'signal'] = -1

        # Clean up temporary column
        df = df.drop(['prev_rsi'], axis=1)

        return df


class CombinedStrategy(BaseStrategy):
    """Combined strategy using multiple indicators"""

    def __init__(self):
        """Initialize combined strategy"""
        super().__init__("Combined Strategy")
        self.ma_strategy = MovingAverageCrossover()
        self.rsi_strategy = RSIStrategy()

    def generate_signals(self, data):
        """
        Generate signals using combination of MA and RSI

        Args:
            data (pd.DataFrame): Price data

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        # Get signals from both strategies
        ma_signals = self.ma_strategy.generate_signals(data)
        rsi_signals = self.rsi_strategy.generate_signals(data)

        # Combine signals (simple approach: average the signals)
        combined_data = data.copy()
        combined_data['ma_signal'] = ma_signals['signal']
        combined_data['rsi_signal'] = rsi_signals['signal']

        # Simple combination: require both strategies to agree
        combined_data['signal'] = 0

        # Both buy signals = buy
        buy_condition = (combined_data['ma_signal'] == 1) & (combined_data['rsi_signal'] == 1)
        combined_data.loc[buy_condition, 'signal'] = 1

        # Both sell signals = sell
        sell_condition = (combined_data['ma_signal'] == -1) & (combined_data['rsi_signal'] == -1)
        combined_data.loc[sell_condition, 'signal'] = -1

        # Clean up temporary columns
        combined_data = combined_data.drop(['ma_signal', 'rsi_signal'], axis=1)

        return combined_data


class MomentumStrategy(BaseStrategy):
    """Enhanced momentum-based trading strategy with better entry/exit timing"""

    def __init__(self, lookback_period=20, confirmation_period=5, momentum_threshold=0.02,
                 volume_threshold=1.2, volatility_filter=0.01, max_holding_period=10):
        """
        Initialize enhanced momentum strategy

        Args:
            lookback_period (int): Period to calculate momentum (increased from 10 to 20)
            confirmation_period (int): Period for signal confirmation
            momentum_threshold (float): Minimum momentum strength to enter
            volume_threshold (float): Volume confirmation multiplier
            volatility_filter (float): Minimum volatility to trade (avoid dead stocks)
            max_holding_period (int): Maximum days to hold position
        """
        super().__init__("Enhanced Momentum Strategy")
        self.lookback_period = lookback_period
        self.confirmation_period = confirmation_period
        self.momentum_threshold = momentum_threshold
        self.volume_threshold = volume_threshold
        self.volatility_filter = volatility_filter
        self.max_holding_period = max_holding_period

    def generate_signals(self, data):
        """
        Generate enhanced momentum-based signals with better entry/exit timing

        Args:
            data (pd.DataFrame): Price data with OHLC columns

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate multiple momentum indicators
        df['momentum_short'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        df['momentum_medium'] = (df['Close'] - df['Close'].shift(self.lookback_period)) / df['Close'].shift(self.lookback_period)
        df['momentum_long'] = (df['Close'] - df['Close'].shift(self.lookback_period * 2)) / df['Close'].shift(self.lookback_period * 2)

        # Calculate price acceleration (momentum of momentum)
        df['momentum_acceleration'] = df['momentum_medium'].diff()

        # Volume confirmation
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        df['volume_trend'] = df['volume_ratio'].rolling(window=5).mean()

        # Volatility filter (avoid trading dead stocks)
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()

        # Price relative strength vs recent range
        df['price_vs_high'] = (df['Close'] - df['High'].rolling(window=self.lookback_period).max()) / df['High'].rolling(window=self.lookback_period).max()
        df['price_vs_low'] = (df['Close'] - df['Low'].rolling(window=self.lookback_period).min()) / df['Low'].rolling(window=self.lookback_period).min()

        # Trend confirmation using moving averages
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()

        # Generate signals with multiple confirmation factors
        df['signal'] = 0

        # Enhanced long conditions (multiple confirmations)
        long_conditions = (
            (df['momentum_medium'] > self.momentum_threshold) &  # Strong medium-term momentum
            (df['momentum_acceleration'] > 0) &  # Accelerating momentum
            (df['volume_ratio'] > self.volume_threshold) &  # Volume confirmation
            (df['volatility'] > self.volatility_filter) &  # Sufficient volatility
            (df['ma_5'] > df['ma_20']) &  # Short-term trend confirmation
            (df['Close'] > df['ma_20']) &  # Price above medium-term trend
            (df['price_vs_high'] > -0.05)  # Not too close to recent highs (avoid exhaustion)
        )

        # Enhanced short conditions
        short_conditions = (
            (df['momentum_medium'] < -self.momentum_threshold) &  # Strong negative momentum
            (df['momentum_acceleration'] < 0) &  # Accelerating decline
            (df['volume_ratio'] > self.volume_threshold) &  # Volume confirmation
            (df['volatility'] > self.volatility_filter) &  # Sufficient volatility
            (df['ma_5'] < df['ma_20']) &  # Short-term trend confirmation
            (df['Close'] < df['ma_20']) &  # Price below medium-term trend
            (df['price_vs_low'] < 0.05)  # Not too close to recent lows (avoid capitulation)
        )

        # Apply signals with confirmation
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1

        # Exit conditions for better timing
        # Exit long when momentum weakens or reverses
        exit_long = (
            (df['signal'] == 1) & (
                (df['momentum_acceleration'] < -0.001) |  # Momentum slowing
                (df['momentum_medium'] < 0) |  # Momentum turned negative
                (df['volume_ratio'] < 0.8) |  # Volume dropping
                (df['Close'] < df['ma_5'])  # Price below short-term trend
            )
        )

        # Exit short when momentum weakens or reverses
        exit_short = (
            (df['signal'] == -1) & (
                (df['momentum_acceleration'] > 0.001) |  # Momentum improving
                (df['momentum_medium'] > 0) |  # Momentum turned positive
                (df['volume_ratio'] < 0.8) |  # Volume dropping
                (df['Close'] > df['ma_5'])  # Price above short-term trend
            )
        )

        df.loc[exit_long | exit_short, 'signal'] = 0

        # Clean up temporary columns
        momentum_columns = ['momentum_short', 'momentum_medium', 'momentum_long', 'momentum_acceleration',
                           'volume_ma', 'volume_ratio', 'volume_trend', 'volatility',
                           'price_vs_high', 'price_vs_low', 'ma_5', 'ma_20', 'ma_50']
        df = df.drop(momentum_columns, axis=1)

        return df


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""

    def __init__(self, lookback_period=20, entry_threshold=1.5, exit_threshold=0.2):
        """
        Initialize mean reversion strategy

        Args:
            lookback_period (int): Period for calculating mean
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
        """
        super().__init__("Mean Reversion Strategy")
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, data):
        """
        Generate mean reversion signals using Bollinger Bands

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate Bollinger Bands
        df['ma'] = df['Close'].rolling(window=self.lookback_period).mean()
        df['std'] = df['Close'].rolling(window=self.lookback_period).std()

        # Calculate z-score (deviation from mean)
        df['z_score'] = (df['Close'] - df['ma']) / df['std']

        # Generate signals
        df['signal'] = 0

        # Buy when price is below mean (oversold) - more responsive
        buy_condition = df['z_score'] < -self.entry_threshold
        df.loc[buy_condition, 'signal'] = 1

        # Sell when price is above mean (overbought)
        sell_condition = df['z_score'] > self.entry_threshold
        df.loc[sell_condition, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['ma', 'std', 'z_score'], axis=1)

        return df


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy"""

    def __init__(self, lookback_period=10, breakout_multiplier=1.2):
        """
        Initialize volatility breakout strategy

        Args:
            lookback_period (int): Period for calculating volatility
            breakout_multiplier (float): Multiplier for breakout threshold
        """
        super().__init__("Volatility Breakout Strategy")
        self.lookback_period = lookback_period
        self.breakout_multiplier = breakout_multiplier

    def generate_signals(self, data):
        """
        Generate signals based on volatility breakouts

        Args:
            data (pd.DataFrame): Price data with OHLC columns

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate average true range (ATR)
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift(1))
        df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.lookback_period).mean()

        # Calculate breakout levels
        df['upper_breakout'] = df['Close'].shift(1) + (df['atr'] * self.breakout_multiplier)
        df['lower_breakout'] = df['Close'].shift(1) - (df['atr'] * self.breakout_multiplier)

        # Generate signals
        df['signal'] = 0

        # Buy breakout: close above upper level or strong upward movement
        buy_condition = (df['Close'] > df['upper_breakout']) | (df['Close'] > df['High'].shift(1))
        df.loc[buy_condition, 'signal'] = 1

        # Sell breakout: close below lower level or strong downward movement
        sell_condition = (df['Close'] < df['lower_breakout']) | (df['Close'] < df['Low'].shift(1))
        df.loc[sell_condition, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['high_low', 'high_close', 'low_close', 'true_range', 'atr', 'upper_breakout', 'lower_breakout'], axis=1)

        return df


class MultiTimeframeStrategy(BaseStrategy):
    """Multi-timeframe strategy combining different timeframes"""

    def __init__(self, short_strategy='ma', long_strategy='momentum'):
        """
        Initialize multi-timeframe strategy

        Args:
            short_strategy (str): Strategy for short-term signals
            long_strategy (str): Strategy for long-term signals
        """
        super().__init__("Multi-Timeframe Strategy")
        self.short_strategy_name = short_strategy
        self.long_strategy_name = long_strategy

        # Initialize strategies
        self.short_strategy = self._get_strategy_instance(short_strategy)
        self.long_strategy = self._get_strategy_instance(long_strategy)

    def _get_strategy_instance(self, strategy_name):
        """Get strategy instance"""
        strategies = {
            'ma': MovingAverageCrossover(),
            'rsi': RSIStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': VolatilityBreakoutStrategy()
        }
        return strategies.get(strategy_name, MovingAverageCrossover())

    def generate_signals(self, data, short_data=None):
        """
        Generate multi-timeframe signals

        Args:
            data (pd.DataFrame): Main timeframe data
            short_data (pd.DataFrame, optional): Shorter timeframe data

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Get short-term signals
        short_signals = self.short_strategy.generate_signals(data)
        df['short_signal'] = short_signals['signal']

        # Get long-term signals
        long_signals = self.long_strategy.generate_signals(data)
        df['long_signal'] = long_signals['signal']

        # Combine signals: require both timeframes to agree
        df['signal'] = 0

        # Both bullish = long
        long_condition = (df['short_signal'] == 1) & (df['long_signal'] == 1)
        df.loc[long_condition, 'signal'] = 1

        # Both bearish = short
        short_condition = (df['short_signal'] == -1) & (df['long_signal'] == -1)
        df.loc[short_condition, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['short_signal', 'long_signal'], axis=1)

        return df


class EnhancedCombinedStrategy(BaseStrategy):
    """Enhanced combined strategy with multiple indicators and voting"""

    def __init__(self, min_votes=1):
        """
        Initialize combined strategy with voting system

        Args:
            min_votes (int): Minimum votes required for a signal
        """
        super().__init__("Enhanced Combined Strategy")
        self.min_votes = min_votes

        # Initialize multiple strategies
        self.strategies = [
            MovingAverageCrossover(),
            RSIStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityBreakoutStrategy()
        ]

    def generate_signals(self, data):
        """
        Generate signals using voting system across multiple strategies

        Args:
            data (pd.DataFrame): Price data

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Get signals from all strategies
        all_signals = []
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                all_signals.append(signals['signal'])
            except Exception as e:
                print(f"Warning: Strategy {strategy.name} failed: {str(e)}")
                continue

        if not all_signals:
            df['signal'] = 0
            return df

        # Create voting matrix
        signals_df = pd.concat(all_signals, axis=1)
        signals_df.columns = [f'strategy_{i}' for i in range(len(all_signals))]

        # Count votes for each signal type
        df['buy_votes'] = (signals_df == 1).sum(axis=1)
        df['sell_votes'] = (signals_df == -1).sum(axis=1)
        df['total_votes'] = df['buy_votes'] + df['sell_votes']

        # Generate final signals based on voting
        df['signal'] = 0

        # Strong buy consensus
        df.loc[(df['buy_votes'] >= self.min_votes) & (df['buy_votes'] > df['sell_votes']), 'signal'] = 1

        # Strong sell consensus
        df.loc[(df['sell_votes'] >= self.min_votes) & (df['sell_votes'] > df['buy_votes']), 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['buy_votes', 'sell_votes', 'total_votes'], axis=1)

        return df


class ScalpingStrategy(BaseStrategy):
    """High-frequency scalping strategy for very short-term trades"""

    def __init__(self, fast_period=2, slow_period=5, min_change=0.0005):
        """
        Initialize scalping strategy

        Args:
            fast_period (int): Fast moving average period
            slow_period (int): Slow moving average period
            min_change (float): Minimum price change to trigger signal
        """
        super().__init__("High-Frequency Scalping")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_change = min_change

    def generate_signals(self, data):
        """
        Generate scalping signals based on very short-term price movements

        Args:
            data (pd.DataFrame): Price data with OHLC columns

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate ultra-short moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_period).mean()

        # Calculate price velocity (rate of change)
        df['price_velocity'] = df['Close'].pct_change(self.fast_period)

        # Generate signals based on rapid crossovers and momentum
        df['signal'] = 0

        # Strong upward momentum + fast MA above slow MA
        strong_up = (df['price_velocity'] > self.min_change) & (df['fast_ma'] > df['slow_ma'])
        df.loc[strong_up, 'signal'] = 1

        # Strong downward momentum + fast MA below slow MA
        strong_down = (df['price_velocity'] < -self.min_change) & (df['fast_ma'] < df['slow_ma'])
        df.loc[strong_down, 'signal'] = -1

        # Also trigger on MA crossovers alone (more sensitive)
        crossover_up = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        df.loc[crossover_up, 'signal'] = 1

        crossover_down = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
        df.loc[crossover_down, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['fast_ma', 'slow_ma', 'price_velocity'], axis=1)

        return df


class ContrarianStrategy(BaseStrategy):
    """Contrarian strategy that trades against the trend (very risky)"""

    def __init__(self, trend_period=10, reversal_threshold=0.02):
        """
        Initialize contrarian strategy

        Args:
            trend_period (int): Period to calculate trend strength
            reversal_threshold (float): How extreme the trend must be before reversal
        """
        super().__init__("Contrarian Reversal")
        self.trend_period = trend_period
        self.reversal_threshold = reversal_threshold

    def generate_signals(self, data):
        """
        Generate signals by betting against strong trends

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate trend strength
        df['trend'] = (df['Close'] - df['Close'].shift(self.trend_period)) / df['Close'].shift(self.trend_period)
        df['trend_strength'] = abs(df['trend'])

        # Calculate overbought/oversold using multiple timeframes
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()

        # Deviation from multiple moving averages
        df['deviation_5'] = (df['Close'] - df['ma_5']) / df['ma_5']
        df['deviation_20'] = (df['Close'] - df['ma_20']) / df['ma_20']

        # Generate contrarian signals
        df['signal'] = 0

        # Bet against strong uptrends (expect reversal) - more sensitive
        extreme_uptrend = (df['trend_strength'] > self.reversal_threshold) & (df['trend'] > 0)
        overbought = (df['deviation_5'] > 0.02) | (df['deviation_20'] > 0.01)
        df.loc[extreme_uptrend & overbought, 'signal'] = -1

        # Bet against strong downtrends (expect reversal) - more sensitive
        extreme_downtrend = (df['trend_strength'] > self.reversal_threshold) & (df['trend'] < 0)
        oversold = (df['deviation_5'] < -0.02) | (df['deviation_20'] < -0.01)
        df.loc[extreme_downtrend & oversold, 'signal'] = 1

        # Also trigger on just extreme deviations (even more sensitive)
        df.loc[df['deviation_5'] > 0.03, 'signal'] = -1  # Moderately overbought
        df.loc[df['deviation_5'] < -0.03, 'signal'] = 1  # Moderately oversold

        # Most sensitive: trigger on any deviation from long-term MA
        df.loc[df['deviation_20'] > 0.02, 'signal'] = -1
        df.loc[df['deviation_20'] < -0.02, 'signal'] = 1

        # Clean up temporary columns
        df = df.drop(['trend', 'trend_strength', 'ma_5', 'ma_20', 'ma_50', 'deviation_5', 'deviation_20'], axis=1)

        return df


class LeveragedMomentumStrategy(BaseStrategy):
    """Leveraged momentum strategy for high-risk, high-reward trading"""

    def __init__(self, momentum_period=5, leverage_factor=3.0, volatility_threshold=0.005):
        """
        Initialize leveraged momentum strategy

        Args:
            momentum_period (int): Period for momentum calculation
            leverage_factor (float): Leverage multiplier (risky!)
            volatility_threshold (float): Minimum volatility to trade
        """
        super().__init__("Leveraged Momentum")
        self.momentum_period = momentum_period
        self.leverage_factor = leverage_factor
        self.volatility_threshold = volatility_threshold

    def generate_signals(self, data):
        """
        Generate leveraged momentum signals with high risk/reward

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate momentum with exponential weighting (recent data more important)
        df['momentum'] = df['Close'].pct_change(self.momentum_period)

        # Calculate volatility for risk filtering
        df['volatility'] = df['Close'].pct_change().rolling(window=self.momentum_period).std()

        # Calculate volume momentum
        df['volume_momentum'] = df['Volume'].pct_change(self.momentum_period)

        # Generate leveraged signals
        df['signal'] = 0

        # Simple momentum-based signals (most sensitive)
        df.loc[df['momentum'] > 0.001, 'signal'] = 1   # Any positive momentum
        df.loc[df['momentum'] < -0.001, 'signal'] = -1  # Any negative momentum

        # Clean up temporary columns
        df = df.drop(['momentum', 'volatility', 'volume_momentum'], axis=1)

        return df


class MachineLearningStyleStrategy(BaseStrategy):
    """Pattern recognition strategy inspired by ML approaches"""

    def __init__(self, pattern_window=10, min_pattern_strength=0.7):
        """
        Initialize ML-style pattern recognition strategy

        Args:
            pattern_window (int): Window for pattern analysis
            min_pattern_strength (float): Minimum pattern strength to act on
        """
        super().__init__("ML-Style Pattern Recognition")
        self.pattern_window = pattern_window
        self.min_pattern_strength = min_pattern_strength

    def generate_signals(self, data):
        """
        Generate signals based on price pattern recognition

        Args:
            data (pd.DataFrame): Price data with OHLC columns

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate multiple technical indicators
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()

        # Price momentum features
        df['mom_1'] = df['Close'].pct_change(1)
        df['mom_3'] = df['Close'].pct_change(3)
        df['mom_5'] = df['Close'].pct_change(5)

        # Volatility features
        df['vol_5'] = df['Close'].pct_change().rolling(5).std()
        df['vol_20'] = df['Close'].pct_change().rolling(20).std()

        # Volume features
        df['volume_ma_5'] = df['Volume'].rolling(5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_5']

        # Generate composite signals based on multiple factors
        df['signal'] = 0

        # Bullish patterns (relaxed conditions)
        bullish_pattern = (
            (df['sma_5'] > df['sma_20']) &
            (df['ema_12'] > df['ema_26']) &
            (df['mom_1'] > -0.001)  # Just not strongly negative
        )
        df.loc[bullish_pattern, 'signal'] = 1

        # Bearish patterns (relaxed conditions)
        bearish_pattern = (
            (df['sma_5'] < df['sma_20']) &
            (df['ema_12'] < df['ema_26']) &
            (df['mom_1'] < 0.001)  # Just not strongly positive
        )
        df.loc[bearish_pattern, 'signal'] = -1

        # Also trigger on just strong momentum indicators
        df.loc[df['mom_1'] > 0.01, 'signal'] = 1  # Strong positive momentum
        df.loc[df['mom_1'] < -0.01, 'signal'] = -1  # Strong negative momentum

        # Clean up temporary columns
        pattern_columns = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'mom_1', 'mom_3', 'mom_5',
                          'vol_5', 'vol_20', 'volume_ma_5', 'volume_ratio']
        df = df.drop(pattern_columns, axis=1)

        return df


class ConservativeTrendStrategy(BaseStrategy):
    """Low-risk, long-term trend following strategy"""

    def __init__(self, trend_period=50, confirmation_period=10, filter_threshold=0.01):
        """
        Initialize conservative trend strategy

        Args:
            trend_period (int): Long-term trend period
            confirmation_period (int): Short-term confirmation period
            filter_threshold (float): Minimum trend strength
        """
        super().__init__("Conservative Trend Following")
        self.trend_period = trend_period
        self.confirmation_period = confirmation_period
        self.filter_threshold = filter_threshold

    def generate_signals(self, data):
        """
        Generate conservative trend-following signals

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Long-term trend
        df['long_trend'] = df['Close'].rolling(window=self.trend_period).mean()
        df['trend_strength'] = (df['Close'] - df['long_trend']) / df['long_trend']

        # Short-term confirmation
        df['short_trend'] = df['Close'].rolling(window=self.confirmation_period).mean()

        # Volume confirmation
        df['volume_trend'] = df['Volume'].rolling(window=self.confirmation_period).mean()

        # Generate conservative signals
        df['signal'] = 0

        # Strong uptrend with confirmation
        strong_uptrend = (
            (df['trend_strength'] > self.filter_threshold) &
            (df['Close'] > df['short_trend']) &
            (df['Volume'] > df['volume_trend'])
        )
        df.loc[strong_uptrend, 'signal'] = 1

        # Strong downtrend with confirmation
        strong_downtrend = (
            (df['trend_strength'] < -self.filter_threshold) &
            (df['Close'] < df['short_trend']) &
            (df['Volume'] > df['volume_trend'])
        )
        df.loc[strong_downtrend, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['long_trend', 'trend_strength', 'short_trend', 'volume_trend'], axis=1)

        return df


class BalancedMultiStrategy(BaseStrategy):
    """Medium-risk balanced strategy combining multiple approaches"""

    def __init__(self, primary_weight=0.4, secondary_weight=0.3, tertiary_weight=0.3):
        """
        Initialize balanced multi-strategy

        Args:
            primary_weight (float): Weight for primary strategy (trend)
            secondary_weight (float): Weight for secondary strategy (momentum)
            tertiary_weight (float): Weight for tertiary strategy (mean reversion)
        """
        super().__init__("Balanced Multi-Strategy")
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        self.tertiary_weight = tertiary_weight

        # Initialize component strategies
        self.trend_strategy = MovingAverageCrossover(short_window=10, long_window=30)
        self.momentum_strategy = MomentumStrategy(lookback_period=15, top_percentile=0.35)
        self.reversion_strategy = MeanReversionStrategy(lookback_period=25, entry_threshold=1.8, exit_threshold=0.3)

    def generate_signals(self, data):
        """
        Generate balanced signals using weighted combination

        Args:
            data (pd.DataFrame): Price data

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Get signals from component strategies
        trend_signals = self.trend_strategy.generate_signals(data)
        momentum_signals = self.momentum_strategy.generate_signals(data)
        reversion_signals = self.reversion_strategy.generate_signals(data)

        # Calculate weighted composite signal
        df['trend_signal'] = trend_signals['signal'] * self.primary_weight
        df['momentum_signal'] = momentum_signals['signal'] * self.secondary_weight
        df['reversion_signal'] = reversion_signals['signal'] * self.tertiary_weight

        # Combine signals
        df['composite_signal'] = df['trend_signal'] + df['momentum_signal'] + df['reversion_signal']

        # Generate final signals
        df['signal'] = 0
        df.loc[df['composite_signal'] > 0.3, 'signal'] = 1
        df.loc[df['composite_signal'] < -0.3, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['trend_signal', 'momentum_signal', 'reversion_signal', 'composite_signal'], axis=1)

        return df


class PairsTradingStrategy(BaseStrategy):
    """Pairs trading strategy using cointegration analysis"""

    def __init__(self, lookback_period=60, entry_threshold=2.0, exit_threshold=0.5,
                 min_half_life=5, max_half_life=50):
        """
        Initialize pairs trading strategy

        Args:
            lookback_period (int): Period for calculating spread statistics
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
            min_half_life (int): Minimum half-life for cointegration
            max_half_life (int): Maximum half-life for cointegration
        """
        super().__init__("Pairs Trading")
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def find_cointegrated_pairs(self, data_dict, min_half_life=5, max_half_life=50):
        """
        Find cointegrated pairs from a dictionary of price data

        Args:
            data_dict (dict): Dictionary with symbols as keys and dataframes as values
            min_half_life (int): Minimum half-life for valid pairs
            max_half_life (int): Maximum half-life for valid pairs

        Returns:
            list: List of cointegrated pairs with their parameters
        """
        symbols = list(data_dict.keys())
        pairs = []

        # Filter to only valid stock symbols (exclude column names)
        valid_symbols = [s for s in symbols if isinstance(s, str) and len(s) <= 10 and s.replace('.', '').replace('-', '').isalnum()]

        for i in range(len(valid_symbols)):
            for j in range(i + 1, len(valid_symbols)):
                symbol1 = valid_symbols[i]
                symbol2 = valid_symbols[j]

                # Get price data
                data1 = data_dict[symbol1]
                data2 = data_dict[symbol2]

                # Find common dates
                common_dates = data1.index.intersection(data2.index)
                if len(common_dates) < self.lookback_period * 2:
                    continue

                try:
                    price1 = data1['Close'].loc[common_dates]
                    price2 = data2['Close'].loc[common_dates]
                except Exception as e:
                    print(f"Warning: Error accessing price data for {symbol1}-{symbol2}: {str(e)}")
                    continue

                # Test for cointegration
                try:
                    # Check if pairs are cointegrated using ADF test
                    hedge_ratio, half_life, is_cointegrated = self._test_cointegration(price1, price2)

                    if is_cointegrated and min_half_life <= half_life <= max_half_life:
                        pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'hedge_ratio': hedge_ratio,
                            'half_life': half_life,
                            'common_dates': common_dates
                        })
                except Exception as e:
                    print(f"Warning: Error testing cointegration for {symbol1}-{symbol2}: {str(e)}")
                    continue

        return pairs

    def _test_cointegration(self, price1, price2):
        """
        Test for cointegration between two price series

        Args:
            price1 (pd.Series): Price series 1
            price2 (pd.Series): Price series 2

        Returns:
            tuple: (hedge_ratio, half_life, is_cointegrated)
        """
        # Calculate hedge ratio using linear regression
        model = stats.linregress(price1, price2)
        hedge_ratio = model.slope

        # Calculate spread
        spread = price2 - hedge_ratio * price1

        # Test for stationarity using ADF test
        adf_result = adfuller(spread)

        # Calculate half-life of mean reversion
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        model = stats.linregress(spread_lag, spread_diff)
        half_life = -np.log(2) / model.slope if model.slope < 0 else float('inf')

        # Check if cointegrated (p-value < 0.05) and has reasonable half-life
        is_cointegrated = adf_result[1] < 0.05 and 1 <= half_life <= 100

        return hedge_ratio, half_life, is_cointegrated

    def generate_signals(self, data_dict, pairs=None):
        """
        Generate pairs trading signals

        Args:
            data_dict (dict): Dictionary with symbols as keys and dataframes as values
            pairs (list, optional): Pre-identified pairs, if None will find them

        Returns:
            dict: Dictionary with symbols as keys and signal dataframes as values
        """
        if pairs is None:
            pairs = self.find_cointegrated_pairs(data_dict)

        if not pairs:
            # Return neutral signals if no pairs found
            return {symbol: pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                   for symbol, data in data_dict.items()}

        # Generate signals for each pair
        all_signals = {}

        for pair in pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            hedge_ratio = pair['hedge_ratio']

            # Get data for this pair
            data1 = data_dict[symbol1]
            data2 = data_dict[symbol2]

            # Find common dates
            common_dates = pair['common_dates']
            common_dates = common_dates.intersection(data1.index).intersection(data2.index)

            if len(common_dates) < self.lookback_period:
                continue

            # Calculate spread and z-score
            price1 = data1['Close'].loc[common_dates]
            price2 = data2['Close'].loc[common_dates]

            spread = price2 - hedge_ratio * price1

            # Calculate spread statistics
            spread_mean = spread.rolling(window=self.lookback_period).mean()
            spread_std = spread.rolling(window=self.lookback_period).std()

            # Calculate z-score
            z_score = (spread - spread_mean) / spread_std

            # Generate signals
            signal1 = pd.Series(0, index=common_dates)
            signal2 = pd.Series(0, index=common_dates)

            # Entry signals
            # When spread is too high (symbol2 overvalued relative to symbol1)
            long_symbol1_condition = z_score < -self.entry_threshold
            short_symbol2_condition = z_score < -self.entry_threshold

            # When spread is too low (symbol2 undervalued relative to symbol1)
            short_symbol1_condition = z_score > self.entry_threshold
            long_symbol2_condition = z_score > self.entry_threshold

            # Apply signals
            signal1.loc[long_symbol1_condition] = 1   # Long symbol1
            signal1.loc[short_symbol1_condition] = -1  # Short symbol1
            signal2.loc[short_symbol2_condition] = -1  # Short symbol2
            signal2.loc[long_symbol2_condition] = 1    # Long symbol2

            # Exit signals (when spread returns to mean)
            exit_long1 = (signal1 == 1) & (abs(z_score) < self.exit_threshold)
            exit_short1 = (signal1 == -1) & (abs(z_score) < self.exit_threshold)
            exit_short2 = (signal2 == -1) & (abs(z_score) < self.exit_threshold)
            exit_long2 = (signal2 == 1) & (abs(z_score) < self.exit_threshold)

            signal1.loc[exit_long1 | exit_short1] = 0
            signal2.loc[exit_short2 | exit_long2] = 0

            # Store signals
            if symbol1 not in all_signals:
                all_signals[symbol1] = pd.DataFrame(index=data1.index)
                all_signals[symbol1]['signal'] = 0

            if symbol2 not in all_signals:
                all_signals[symbol2] = pd.DataFrame(index=data2.index)
                all_signals[symbol2]['signal'] = 0

            # Update signals
            all_signals[symbol1].loc[common_dates, 'signal'] = signal1.values
            all_signals[symbol2].loc[common_dates, 'signal'] = signal2.values

        # Fill any missing symbols with neutral signals
        for symbol in data_dict.keys():
            if symbol not in all_signals:
                all_signals[symbol] = pd.DataFrame({'signal': [0] * len(data_dict[symbol])},
                                                   index=data_dict[symbol].index)

        return all_signals


class StatisticalArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage using multiple mean-reverting pairs"""

    def __init__(self, lookback_period=60, entry_threshold=1.5, exit_threshold=0.3,
                 max_positions=5, min_correlation=0.7):
        """
        Initialize statistical arbitrage strategy

        Args:
            lookback_period (int): Period for calculating statistics
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
            max_positions (int): Maximum number of simultaneous positions
            min_correlation (float): Minimum correlation for pair consideration
        """
        super().__init__("Statistical Arbitrage")
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_positions = max_positions
        self.min_correlation = min_correlation
        self.pairs_trading = PairsTradingStrategy(lookback_period, entry_threshold, exit_threshold)

    def generate_signals(self, data_dict):
        """
        Generate statistical arbitrage signals across multiple pairs

        Args:
            data_dict (dict): Dictionary with symbols as keys and dataframes as values

        Returns:
            dict: Dictionary with symbols as keys and signal dataframes as values
        """
        # Find all cointegrated pairs
        pairs = self.pairs_trading.find_cointegrated_pairs(data_dict)

        if not pairs:
            # Return neutral signals if no pairs found
            return {symbol: pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                   for symbol, data in data_dict.items()}

        # Sort pairs by half-life (prefer faster mean reversion)
        pairs.sort(key=lambda x: x['half_life'])

        # Generate signals for each pair
        pair_signals = {}
        for pair in pairs[:self.max_positions]:  # Limit to max_positions pairs
            signals = self.pairs_trading.generate_signals(data_dict, [pair])
            pair_signals[pair['symbol1']] = signals[pair['symbol1']]
            pair_signals[pair['symbol2']] = signals[pair['symbol2']]

        # Combine signals for each symbol (simple averaging)
        final_signals = {}
        for symbol in data_dict.keys():
            if symbol in pair_signals:
                final_signals[symbol] = pair_signals[symbol].copy()
            else:
                final_signals[symbol] = pd.DataFrame({'signal': [0] * len(data_dict[symbol])},
                                                     index=data_dict[symbol].index)

        return final_signals


class SectorRotationStrategy(BaseStrategy):
    """Sector rotation strategy based on relative strength"""

    def __init__(self, lookback_period=60, top_sectors=3, rotation_threshold=0.1):
        """
        Initialize sector rotation strategy

        Args:
            lookback_period (int): Period for calculating relative strength
            top_sectors (int): Number of top sectors to hold
            rotation_threshold (float): Minimum strength difference for rotation
        """
        super().__init__("Sector Rotation")
        self.lookback_period = lookback_period
        self.top_sectors = top_sectors
        self.rotation_threshold = rotation_threshold

        # Define sector ETFs/major stocks for each sector
        self.sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRNA', 'ABT', 'TMO'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC'],
            'Consumer': ['WMT', 'HD', 'MCD', 'DIS', 'NKE', 'KO', 'PEP'],
            'Industrial': ['BA', 'CAT', 'GE', 'UPS', 'HON', 'LMT'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW'],
            'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'EXC'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG'],
            'Communication': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ']
        }

    def calculate_sector_strength(self, data_dict):
        """
        Calculate relative strength for each sector

        Args:
            data_dict (dict): Dictionary with symbols as keys and dataframes as values

        Returns:
            dict: Sector strength scores
        """
        sector_strength = {}

        for sector, symbols in self.sector_mapping.items():
            sector_symbols = [s for s in symbols if s in data_dict]

            if not sector_symbols:
                continue

            # Calculate average momentum for the sector
            sector_momentum = []
            for symbol in sector_symbols:
                if symbol in data_dict:
                    data = data_dict[symbol]
                    if len(data) >= self.lookback_period:
                        # Simple momentum: price change over lookback period
                        momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-self.lookback_period]) / data['Close'].iloc[-self.lookback_period]
                        sector_momentum.append(momentum)

            if sector_momentum:
                sector_strength[sector] = np.mean(sector_momentum)

        return sector_strength

    def generate_signals(self, data_dict):
        """
        Generate sector rotation signals

        Args:
            data_dict (dict): Dictionary with symbols as keys and dataframes as values

        Returns:
            dict: Dictionary with symbols as keys and signal dataframes as values
        """
        # Calculate sector strength
        sector_strength = self.calculate_sector_strength(data_dict)

        if not sector_strength:
            return {symbol: pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
                   for symbol, data in data_dict.items()}

        # Sort sectors by strength
        sorted_sectors = sorted(sector_strength.items(), key=lambda x: x[1], reverse=True)

        # Get top sectors
        top_sectors_list = [sector for sector, strength in sorted_sectors[:self.top_sectors]]

        # Generate signals for symbols in top sectors
        signals = {}
        for symbol in data_dict.keys():
            data = data_dict[symbol]
            signals[symbol] = pd.DataFrame({'signal': [0] * len(data)}, index=data.index)

            # Check which sector this symbol belongs to
            symbol_sector = None
            for sector, symbols in self.sector_mapping.items():
                if symbol in symbols:
                    symbol_sector = sector
                    break

            if symbol_sector and symbol_sector in top_sectors_list:
                # Long signal for symbols in top sectors
                signals[symbol]['signal'] = 1

        return signals


class MarketRegimeStrategy(BaseStrategy):
    """Market regime detection and adaptive strategy selection"""

    def __init__(self, regime_lookback=60, trend_threshold=0.1, volatility_threshold=0.02):
        """
        Initialize market regime strategy

        Args:
            regime_lookback (int): Period for regime detection
            trend_threshold (float): Threshold for trend identification
            volatility_threshold (float): Threshold for volatility classification
        """
        super().__init__("Market Regime Adaptive")
        self.regime_lookback = regime_lookback
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold

        # Initialize component strategies for different regimes
        self.trend_strategy = MovingAverageCrossover(short_window=10, long_window=50)
        self.mean_reversion_strategy = MeanReversionStrategy(lookback_period=30)
        self.momentum_strategy = MomentumStrategy(lookback_period=20)

    def detect_market_regime(self, data):
        """
        Detect current market regime (trending, mean-reverting, or random)

        Args:
            data (pd.DataFrame or pd.Series): Price data

        Returns:
            str: Market regime ('trending', 'mean_reverting', 'high_volatility')
        """
        if len(data) < self.regime_lookback:
            return 'mean_reverting'  # Default for insufficient data

        # Handle both DataFrame and Series inputs
        if isinstance(data, pd.DataFrame):
            prices = data['Close']
        else:
            # data is already a Series (Close prices)
            prices = data

        # Calculate trend strength
        trend_strength = abs(prices.iloc[-1] - prices.iloc[-self.regime_lookback]) / prices.iloc[-self.regime_lookback]

        # Calculate volatility
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.regime_lookback).std().iloc[-1]

        # Classify regime
        if trend_strength > self.trend_threshold and volatility < self.volatility_threshold:
            return 'trending'
        elif volatility > self.volatility_threshold * 2:
            return 'high_volatility'
        else:
            return 'mean_reverting'

    def generate_signals(self, data):
        """
        Generate adaptive signals based on market regime

        Args:
            data (pd.DataFrame or dict): Either a single DataFrame or dictionary with symbols as keys and dataframes as values

        Returns:
            pd.DataFrame or dict: Either a single DataFrame with signals or dictionary with symbols as keys and signal dataframes as values
        """
        # Handle both single DataFrame and dictionary inputs
        if isinstance(data, dict):
            # Multi-symbol case
            signals = {}
            for symbol, symbol_data in data.items():
                if len(symbol_data) < self.regime_lookback:
                    signals[symbol] = pd.DataFrame({'signal': [0] * len(symbol_data)}, index=symbol_data.index)
                    continue

                # Detect regime for this symbol
                regime = self.detect_market_regime(symbol_data)

                # Select appropriate strategy based on regime
                if regime == 'trending':
                    strategy_signals = self.trend_strategy.generate_signals(symbol_data)
                elif regime == 'high_volatility':
                    strategy_signals = self.momentum_strategy.generate_signals(symbol_data)
                else:  # mean_reverting
                    strategy_signals = self.mean_reversion_strategy.generate_signals(symbol_data)

                signals[symbol] = strategy_signals

            return signals
        else:
            # Single-symbol case
            if len(data) < self.regime_lookback:
                return pd.DataFrame({'signal': [0] * len(data)}, index=data.index)

            # Detect regime for this symbol
            regime = self.detect_market_regime(data)

            # Select appropriate strategy based on regime
            if regime == 'trending':
                return self.trend_strategy.generate_signals(data)
            elif regime == 'high_volatility':
                return self.momentum_strategy.generate_signals(data)
            else:  # mean_reverting
                return self.mean_reversion_strategy.generate_signals(data)
