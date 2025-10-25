"""
Simple trading strategies for algorithmic trading
"""
import pandas as pd
import numpy as np


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

    def __init__(self, short_window=10, long_window=50):
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
        df['prev_ma_short'] = df['ma_short'].shift(1)
        df['prev_ma_long'] = df['ma_long'].shift(1)

        # Buy signal: short MA crosses above long MA
        buy_condition = (df['prev_ma_short'] <= df['prev_ma_long']) & (df['ma_short'] > df['ma_long'])
        df.loc[buy_condition, 'signal'] = 1

        # Sell signal: short MA crosses below long MA
        sell_condition = (df['prev_ma_short'] >= df['prev_ma_long']) & (df['ma_short'] < df['ma_long'])
        df.loc[sell_condition, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['prev_ma_short', 'prev_ma_long'], axis=1)

        return df


class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) Strategy"""

    def __init__(self, period=14, overbought=70, oversold=30):
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

        # Buy signal: RSI crosses above oversold level
        df['prev_rsi'] = df['rsi'].shift(1)
        buy_condition = (df['prev_rsi'] <= self.oversold) & (df['rsi'] > self.oversold)
        df.loc[buy_condition, 'signal'] = 1

        # Sell signal: RSI crosses below overbought level
        sell_condition = (df['prev_rsi'] >= self.overbought) & (df['rsi'] < self.overbought)
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
    """Momentum-based trading strategy"""

    def __init__(self, lookback_period=20, hold_period=5, top_percentile=0.3):
        """
        Initialize momentum strategy

        Args:
            lookback_period (int): Period to calculate momentum
            hold_period (int): How long to hold positions
            top_percentile (float): Top percentile to select for long positions
        """
        super().__init__("Momentum Strategy")
        self.lookback_period = lookback_period
        self.hold_period = hold_period
        self.top_percentile = top_percentile

    def generate_signals(self, data):
        """
        Generate momentum-based signals

        Args:
            data (pd.DataFrame): Price data with 'Close' column

        Returns:
            pd.DataFrame: Data with 'signal' column (-1, 0, 1)
        """
        df = data.copy()

        # Calculate momentum (rate of change)
        df['momentum'] = (df['Close'] - df['Close'].shift(self.lookback_period)) / df['Close'].shift(self.lookback_period)

        # Calculate relative strength vs market (using volume as proxy)
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['relative_strength'] = df['momentum'] * (df['Volume'] / df['volume_ma'])

        # Generate signals based on momentum ranking
        df['signal'] = 0
        df['momentum_rank'] = df['momentum'].rolling(window=self.lookback_period).rank(pct=True)

        # Long top performers
        long_condition = df['momentum_rank'] > (1 - self.top_percentile)
        df.loc[long_condition, 'signal'] = 1

        # Short bottom performers
        short_condition = df['momentum_rank'] < self.top_percentile
        df.loc[short_condition, 'signal'] = -1

        # Clean up temporary columns
        df = df.drop(['momentum', 'volume_ma', 'relative_strength', 'momentum_rank'], axis=1)

        return df


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""

    def __init__(self, lookback_period=20, entry_threshold=2.0, exit_threshold=0.5):
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
        df['prev_z_score'] = df['z_score'].shift(1)

        # Buy when price is significantly below mean (oversold)
        buy_condition = (df['prev_z_score'] < -self.entry_threshold) & (df['z_score'] >= -self.entry_threshold)
        df.loc[buy_condition, 'signal'] = 1

        # Sell when price is significantly above mean (overbought)
        sell_condition = (df['prev_z_score'] > self.entry_threshold) & (df['z_score'] <= self.entry_threshold)
        df.loc[sell_condition, 'signal'] = -1

        # Exit positions when back to mean
        exit_long_condition = (df['prev_z_score'] < self.exit_threshold) & (df['z_score'] >= self.exit_threshold)
        exit_short_condition = (df['prev_z_score'] > -self.exit_threshold) & (df['z_score'] <= -self.exit_threshold)

        df.loc[exit_long_condition & (df['signal'] == 0), 'signal'] = -1  # Close long
        df.loc[exit_short_condition & (df['signal'] == 0), 'signal'] = 1   # Close short

        # Clean up temporary columns
        df = df.drop(['ma', 'std', 'z_score', 'prev_z_score'], axis=1)

        return df


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy"""

    def __init__(self, lookback_period=20, breakout_multiplier=1.5):
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

        # Buy breakout: close above upper level
        buy_condition = df['Close'] > df['upper_breakout']
        df.loc[buy_condition, 'signal'] = 1

        # Sell breakout: close below lower level
        sell_condition = df['Close'] < df['lower_breakout']
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

    def __init__(self, min_votes=2):
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
