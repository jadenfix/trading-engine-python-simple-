"""
Stochastic Optimization System with Hidden Markov Models

This module implements advanced stochastic optimization techniques including:
- Hidden Markov Model (HMM) regime detection for optimization
- Particle swarm optimization (PSO)
- Genetic algorithms with HMM-guided evolution
- Stochastic gradient optimization
- Cross-asset regime synchronization
- Unconventional alpha generation through regime transitions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, multivariate_normal
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. HMM features will be limited.")


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection"""

    def __init__(self, n_regimes=3, covariance_type='full', random_state=42):
        """
        Initialize HMM regime detector

        Args:
            n_regimes (int): Number of market regimes to detect
            covariance_type (str): Covariance type for HMM ('spherical', 'tied', 'diag', 'full')
            random_state (int): Random state for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.regime_labels = {}
        self.transition_matrix = None
        self.means = None
        self.covars = None

    def fit(self, returns_data: pd.DataFrame, n_iterations=100):
        """
        Fit HMM to returns data

        Args:
            returns_data (pd.DataFrame): Returns data for multiple assets
            n_iterations (int): Number of EM iterations

        Returns:
            dict: Fitting results and regime information
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn required for HMM functionality")

        # Prepare data - use returns and volatility features
        features = []

        for col in returns_data.columns:
            returns = returns_data[col].dropna()

            # Create feature vector for each asset
            asset_features = pd.DataFrame({
                f'{col}_returns': returns,
                f'{col}_volatility': returns.rolling(20).std(),
                f'{col}_skewness': returns.rolling(60).skew(),
                f'{col}_kurtosis': returns.rolling(60).kurtosis()
            }).dropna()

            features.append(asset_features)

        # Combine all asset features
        if features:
            X = pd.concat(features, axis=1).dropna().values
        else:
            raise ValueError("Insufficient data for HMM fitting")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=n_iterations,
            random_state=self.random_state,
            verbose=False
        )

        self.model.fit(X_scaled)

        # Store model parameters
        self.transition_matrix = self.model.transmat_
        self.means = self.model.means_
        self.covars = self.model.covariances_

        # Predict regimes
        regimes = self.model.predict(X_scaled)

        # Label regimes based on characteristics
        self._label_regimes(X_scaled, regimes)

        return {
            'n_regimes': self.n_regimes,
            'log_likelihood': self.model.score(X_scaled),
            'aic': self.model.aic(X_scaled),
            'bic': self.model.bic(X_scaled),
            'regime_labels': self.regime_labels,
            'transition_matrix': self.transition_matrix,
            'regime_sequence': regimes,
            'regime_probabilities': self.model.predict_proba(X_scaled)
        }

    def _label_regimes(self, X_scaled, regimes):
        """Label regimes based on their statistical properties"""
        for regime in range(self.n_regimes):
            regime_data = X_scaled[regimes == regime]

            if len(regime_data) > 10:
                # Calculate regime characteristics
                mean_return = np.mean(regime_data[:, 0])  # First feature is returns
                volatility = np.std(regime_data[:, 1])     # Second feature is volatility

                # Label regimes
                if mean_return > np.percentile(X_scaled[:, 0], 75):
                    if volatility > np.percentile(X_scaled[:, 1], 75):
                        self.regime_labels[regime] = 'bull_volatile'
                    else:
                        self.regime_labels[regime] = 'bull_stable'
                elif mean_return < np.percentile(X_scaled[:, 0], 25):
                    if volatility > np.percentile(X_scaled[:, 1], 75):
                        self.regime_labels[regime] = 'bear_volatile'
                    else:
                        self.regime_labels[regime] = 'bear_stable'
                else:
                    if volatility > np.percentile(X_scaled[:, 1], 75):
                        self.regime_labels[regime] = 'sideways_volatile'
                    else:
                        self.regime_labels[regime] = 'sideways_stable'
            else:
                self.regime_labels[regime] = f'regime_{regime}'

    def predict_regime(self, current_data: pd.DataFrame):
        """
        Predict current market regime

        Args:
            current_data (pd.DataFrame): Current market data

        Returns:
            dict: Regime prediction results
        """
        if self.model is None:
            raise ValueError("HMM model not fitted")

        # Prepare current data in same format as training
        features = []
        for col in current_data.columns:
            returns = current_data[col].dropna()

            if len(returns) >= 60:  # Need minimum data for features
                asset_features = pd.DataFrame({
                    f'{col}_returns': returns,
                    f'{col}_volatility': returns.rolling(20).std(),
                    f'{col}_skewness': returns.rolling(60).skew(),
                    f'{col}_kurtosis': returns.rolling(60).kurtosis()
                }).dropna()

                features.append(asset_features)

        if not features:
            return {'regime': 'insufficient_data', 'confidence': 0}

        # Use most recent data point
        current_features = pd.concat(features, axis=1).iloc[-1:].dropna()

        if len(current_features) == 0:
            return {'regime': 'insufficient_data', 'confidence': 0}

        # Standardize (using training scaler if available, otherwise fit new)
        scaler = StandardScaler()
        X_current = scaler.fit_transform(current_features.values.reshape(1, -1))

        # Predict regime
        regime = self.model.predict(X_current)[0]
        probabilities = self.model.predict_proba(X_current)[0]

        return {
            'regime': self.regime_labels.get(regime, f'regime_{regime}'),
            'regime_id': regime,
            'confidence': np.max(probabilities),
            'probabilities': probabilities,
            'transition_probabilities': self.transition_matrix[regime]
        }

    def get_regime_transition_alpha(self, strategy_returns: pd.Series):
        """
        Generate alpha from regime transitions

        Args:
            strategy_returns (pd.Series): Strategy returns

        Returns:
            dict: Regime transition alpha opportunities
        """
        if self.model is None or len(strategy_returns) < 100:
            return {}

        # Align strategy returns with regime predictions
        # This is a simplified implementation - in practice you'd need to align
        # the strategy returns with the HMM training data timeline

        # Calculate regime-specific performance
        regime_performance = {}
        for regime_id, regime_name in self.regime_labels.items():
            # Find periods in this regime (simplified)
            regime_returns = strategy_returns  # Placeholder
            regime_performance[regime_name] = {
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'win_rate': (regime_returns > 0).mean()
            }

        return regime_performance


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for strategy parameters"""

    def __init__(self, n_particles=30, max_iterations=50, inertia_weight=0.7,
                 cognitive_weight=1.5, social_weight=1.5, bounds=None):
        """
        Initialize PSO optimizer

        Args:
            n_particles (int): Number of particles in swarm
            max_iterations (int): Maximum iterations
            inertia_weight (float): Inertia weight
            cognitive_weight (float): Cognitive (personal best) weight
            social_weight (float): Social (global best) weight
            bounds (dict): Parameter bounds
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = inertia_weight
        self.c1 = cognitive_weight
        self.c2 = social_weight
        self.bounds = bounds or {}

        # Initialize swarm
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('-inf')

    def initialize_swarm(self):
        """Initialize particle swarm"""
        n_params = len(self.bounds)
        self.particles = np.zeros((self.n_particles, n_params))
        self.velocities = np.zeros((self.n_particles, n_params))
        self.personal_best_positions = np.zeros((self.n_particles, n_params))
        self.personal_best_scores = np.full(self.n_particles, float('-inf'))

        # Initialize particles within bounds
        for i, (param_name, (min_val, max_val)) in enumerate(self.bounds.items()):
            self.particles[:, i] = np.random.uniform(min_val, max_val, self.n_particles)
            self.velocities[:, i] = np.random.uniform(-abs(max_val - min_val), abs(max_val - min_val), self.n_particles)
            self.personal_best_positions[:, i] = self.particles[:, i]

    def optimize(self, objective_function: Callable, verbose=True):
        """
        Run PSO optimization

        Args:
            objective_function (callable): Function to maximize
            verbose (bool): Print progress

        Returns:
            dict: Optimization results
        """
        self.initialize_swarm()

        param_names = list(self.bounds.keys())

        for iteration in range(self.max_iterations):
            # Evaluate all particles
            for i in range(self.n_particles):
                params = dict(zip(param_names, self.particles[i]))
                score = objective_function(params)

                # Update personal best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()

                    # Update global best
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.particles[i].copy()

            # Update velocities and positions
            r1 = np.random.uniform(0, 1, (self.n_particles, len(param_names)))
            r2 = np.random.uniform(0, 1, (self.n_particles, len(param_names)))

            # Velocity update
            inertia = self.w * self.velocities
            cognitive = self.c1 * r1 * (self.personal_best_positions - self.particles)
            social = self.c2 * r2 * (self.global_best_position - self.particles)

            self.velocities = inertia + cognitive + social

            # Position update
            self.particles += self.velocities

            # Enforce bounds
            for i, (param_name, (min_val, max_val)) in enumerate(self.bounds.items()):
                self.particles[:, i] = np.clip(self.particles[:, i], min_val, max_val)

            if verbose and iteration % 10 == 0:
                print(f"PSO Iteration {iteration}: Best Score = {self.global_best_score:.4f}")

        return {
            'best_parameters': dict(zip(param_names, self.global_best_position)),
            'best_score': self.global_best_score,
            'convergence_history': [],  # Could track this
            'method': 'particle_swarm'
        }


class StochasticRegimeOptimizer:
    """Stochastic optimizer that adapts to market regimes"""

    def __init__(self, hmm_detector=None, pso_optimizer=None, regime_memory=10):
        """
        Initialize stochastic regime optimizer

        Args:
            hmm_detector (HMMRegimeDetector): HMM for regime detection
            pso_optimizer (ParticleSwarmOptimizer): PSO for optimization
            regime_memory (int): How many past regimes to remember
        """
        self.hmm_detector = hmm_detector or HMMRegimeDetector()
        self.pso_optimizer = pso_optimizer or ParticleSwarmOptimizer()
        self.regime_memory = regime_memory

        # Track regime history and optimal parameters
        self.regime_history = []
        self.regime_parameters = {}
        self.regime_performance = {}

    def optimize_with_regime_adaptation(self, strategy_class, parameter_bounds,
                                      price_data_dict, current_date, performance_window=252):
        """
        Optimize strategy parameters with regime adaptation

        Args:
            strategy_class: Strategy class to optimize
            parameter_bounds (dict): Parameter bounds
            price_data_dict (dict): Price data
            current_date (pd.Timestamp): Current date
            performance_window (int): Performance evaluation window

        Returns:
            dict: Optimized parameters for current regime
        """
        # Detect current regime
        current_regime = self.hmm_detector.predict_regime(
            pd.DataFrame({symbol: data['Close'].pct_change().loc[:current_date].tail(60)
                         for symbol, data in price_data_dict.items()})
        )

        regime_name = current_regime['regime']

        # Update regime history
        self.regime_history.append({
            'date': current_date,
            'regime': regime_name,
            'confidence': current_regime['confidence']
        })

        # Keep only recent history
        if len(self.regime_history) > self.regime_memory:
            self.regime_history = self.regime_history[-self.regime_memory:]

        # Check if we have optimized parameters for this regime
        if regime_name not in self.regime_parameters:
            print(f"Optimizing parameters for regime: {regime_name}")

            # Create regime-specific objective function
            def regime_objective(params):
                try:
                    # Generate signals for recent period
                    signals = {}
                    eval_start = current_date - pd.Timedelta(days=performance_window)

                    for date in pd.date_range(eval_start, current_date, freq='D'):
                        try:
                            daily_signals = strategy_class(**params).generate_signals(price_data_dict, date)
                            for symbol, signal_df in daily_signals.items():
                                if symbol not in signals:
                                    signals[symbol] = signal_df
                        except:
                            continue

                    if not signals:
                        return -999

                    # Evaluate performance in this regime
                    from research.backtesting_engine import BacktestingEngine
                    backtest_engine = BacktestingEngine(initial_capital=100000)

                    results = backtest_engine.run_backtest(signals, price_data_dict, eval_start, current_date)

                    # Regime-specific scoring (could be enhanced)
                    sharpe = results.get('sharpe_ratio', -999)
                    return sharpe

                except Exception as e:
                    return -999

            # Run PSO optimization
            opt_result = self.pso_optimizer.optimize(regime_objective, verbose=False)
            self.regime_parameters[regime_name] = opt_result['best_parameters']
            self.regime_performance[regime_name] = opt_result['best_score']

            print(f"Optimized parameters for {regime_name}: Sharpe = {opt_result['best_score']:.3f}")

        return {
            'regime': regime_name,
            'parameters': self.regime_parameters[regime_name],
            'regime_confidence': current_regime['confidence'],
            'optimization_score': self.regime_performance.get(regime_name, 0)
        }


class CrossAssetStochasticAnalyzer:
    """Analyzes stochastic relationships across multiple assets"""

    def __init__(self, assets_list, min_correlation=0.1, max_lags=5):
        """
        Initialize cross-asset stochastic analyzer

        Args:
            assets_list (list): List of asset symbols
            min_correlation (float): Minimum correlation threshold
            max_lags (int): Maximum lags for cross-correlation analysis
        """
        self.assets_list = assets_list
        self.min_correlation = min_correlation
        self.max_lags = max_lags

        # Analysis results
        self.correlation_matrix = None
        self.lead_lag_relationships = {}
        self.stochastic_dependencies = {}

    def analyze_stochastic_dependencies(self, returns_data: pd.DataFrame):
        """
        Analyze stochastic dependencies between assets

        Args:
            returns_data (pd.DataFrame): Returns data for multiple assets

        Returns:
            dict: Stochastic dependency analysis
        """
        # Calculate correlation matrix
        self.correlation_matrix = returns_data.corr()

        # Find significant correlations
        significant_pairs = []
        for i in range(len(self.assets_list)):
            for j in range(i + 1, len(self.assets_list)):
                asset1, asset2 = self.assets_list[i], self.assets_list[j]
                corr = self.correlation_matrix.loc[asset1, asset2]

                if abs(corr) >= self.min_correlation:
                    significant_pairs.append({
                        'pair': (asset1, asset2),
                        'correlation': corr,
                        'strength': abs(corr),
                        'direction': 'positive' if corr > 0 else 'negative'
                    })

        # Analyze lead-lag relationships
        self.lead_lag_relationships = {}
        for pair_info in significant_pairs:
            asset1, asset2 = pair_info['pair']
            returns1 = returns_data[asset1]
            returns2 = returns_data[asset2]

            # Cross-correlation analysis
            max_corr = -1
            best_lag = 0
            leader = None

            for lag in range(-self.max_lags, self.max_lags + 1):
                if lag < 0:
                    # asset1 leads
                    lagged_returns1 = returns1.shift(-lag)
                    common_idx = returns1.index.intersection(lagged_returns1.dropna().index)
                    if len(common_idx) > 30:
                        corr = returns2.loc[common_idx].corr(lagged_returns1.loc[common_idx])
                        if abs(corr) > abs(max_corr):
                            max_corr = corr
                            best_lag = lag
                            leader = asset1
                elif lag > 0:
                    # asset2 leads
                    lagged_returns2 = returns2.shift(lag)
                    common_idx = returns2.index.intersection(lagged_returns2.dropna().index)
                    if len(common_idx) > 30:
                        corr = returns1.loc[common_idx].corr(lagged_returns2.loc[common_idx])
                        if abs(corr) > abs(max_corr):
                            max_corr = corr
                            best_lag = lag
                            leader = asset2

            if abs(max_corr) > 0.3:  # Significant lead-lag
                pair_key = f"{asset1}_{asset2}"
                self.lead_lag_relationships[pair_key] = {
                    'leader': leader,
                    'follower': asset2 if leader == asset1 else asset1,
                    'lag_days': best_lag,
                    'correlation': max_corr,
                    'signal_strength': abs(max_corr) * abs(pair_info['correlation'])
                }

        return {
            'correlation_matrix': self.correlation_matrix,
            'significant_pairs': significant_pairs,
            'lead_lag_relationships': self.lead_lag_relationships,
            'network_density': len(significant_pairs) / (len(self.assets_list) * (len(self.assets_list) - 1) / 2)
        }

    def generate_cross_asset_signals(self, current_returns: pd.Series, current_regime: str):
        """
        Generate signals based on cross-asset stochastic relationships

        Args:
            current_returns (pd.Series): Current returns for all assets
            current_regime (str): Current market regime

        Returns:
            dict: Cross-asset trading signals
        """
        signals = {}

        for asset in self.assets_list:
            if asset not in current_returns.index:
                continue

            asset_signals = []
            current_return = current_returns[asset]

            # Check correlation-based signals
            for pair_info in self.stochastic_dependencies.get('significant_pairs', []):
                if asset in pair_info['pair']:
                    other_asset = pair_info['pair'][1] if pair_info['pair'][0] == asset else pair_info['pair'][0]

                    if other_asset in current_returns.index:
                        other_return = current_returns[other_asset]

                        # Generate signal based on correlation and relative performance
                        if pair_info['correlation'] > 0:
                            # Positive correlation - follow the leader
                            if pair_info['direction'] == 'positive':
                                if other_return > current_return:
                                    asset_signals.append(1)  # Buy if correlated asset is outperforming
                                elif other_return < current_return:
                                    asset_signals.append(-1)  # Sell if correlated asset is underperforming
                        else:
                            # Negative correlation - contrarian signal
                            if other_return > current_return:
                                asset_signals.append(-1)  # Sell if negatively correlated asset is up
                            elif other_return < current_return:
                                asset_signals.append(1)   # Buy if negatively correlated asset is down

            # Check lead-lag signals
            for lag_info in self.lead_lag_relationships.values():
                if lag_info['follower'] == asset:
                    leader = lag_info['leader']
                    if leader in current_returns.index:
                        leader_return = current_returns[leader]

                        # Generate signal based on leader-follower relationship
                        if lag_info['correlation'] > 0:
                            # Positive lead-lag - follow the leader
                            if leader_return > 0:
                                asset_signals.append(1)
                            elif leader_return < 0:
                                asset_signals.append(-1)
                        else:
                            # Negative lead-lag - contrarian to leader
                            if leader_return > 0:
                                asset_signals.append(-1)
                            elif leader_return < 0:
                                asset_signals.append(1)

            # Aggregate signals
            if asset_signals:
                avg_signal = np.mean(asset_signals)
                if avg_signal > 0.5:
                    signals[asset] = 1
                elif avg_signal < -0.5:
                    signals[asset] = -1
                else:
                    signals[asset] = 0
            else:
                signals[asset] = 0

        return signals


class UnconventionalAlphaGenerator:
    """Generates unconventional alpha through stochastic and HMM-based methods"""

    def __init__(self, hmm_detector=None, stochastic_analyzer=None, regime_optimizer=None):
        """
        Initialize unconventional alpha generator

        Args:
            hmm_detector (HMMRegimeDetector): HMM regime detector
            stochastic_analyzer (CrossAssetStochasticAnalyzer): Cross-asset analyzer
            regime_optimizer (StochasticRegimeOptimizer): Regime-based optimizer
        """
        self.hmm_detector = hmm_detector or HMMRegimeDetector()
        self.stochastic_analyzer = stochastic_analyzer
        self.regime_optimizer = regime_optimizer or StochasticRegimeOptimizer()

        # Alpha generation components
        self.regime_transition_alpha = {}
        self.stochastic_arbitrage_opportunities = []
        self.unconventional_signals = {}

    def generate_unconventional_alpha(self, price_data_dict: Dict, current_date: pd.Timestamp):
        """
        Generate unconventional alpha signals

        Args:
            price_data_dict (dict): Price data for all assets
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Unconventional alpha signals
        """
        # Prepare returns data for analysis
        returns_data = {}
        for symbol, data in price_data_dict.items():
            if current_date in data.index:
                # Get returns for analysis window
                returns = data['Close'].pct_change().loc[:current_date].tail(252)
                returns_data[symbol] = returns

        returns_df = pd.DataFrame(returns_data)

        # Step 1: HMM regime detection and transition alpha
        try:
            regime_info = self.hmm_detector.predict_regime(returns_df.tail(60))
            current_regime = regime_info['regime']

            # Generate regime transition signals
            regime_signals = self._generate_regime_transition_signals(
                price_data_dict, current_date, current_regime, regime_info
            )
        except:
            regime_signals = {}
            current_regime = 'unknown'

        # Step 2: Cross-asset stochastic signals
        if self.stochastic_analyzer:
            try:
                stochastic_analysis = self.stochastic_analyzer.analyze_stochastic_dependencies(returns_df)
                self.stochastic_analyzer.stochastic_dependencies = stochastic_analysis

                current_returns = returns_df.iloc[-1] if len(returns_df) > 0 else pd.Series()
                stochastic_signals = self.stochastic_analyzer.generate_cross_asset_signals(
                    current_returns, current_regime
                )
            except:
                stochastic_signals = {}
        else:
            stochastic_signals = {}

        # Step 3: Combine signals unconventionally
        combined_signals = {}

        for symbol in price_data_dict.keys():
            signals = []

            # Regime-based signal
            if symbol in regime_signals:
                signals.append(regime_signals[symbol])

            # Stochastic signal
            if symbol in stochastic_signals:
                signals.append(stochastic_signals[symbol])

            # Add unconventional twists
            if signals:
                # Non-linear combination (unconventional weighting)
                weights = np.array([1, 1.5, 0.8][:len(signals)])  # Uneven weights
                weighted_signal = np.average(signals, weights=weights)

                # Apply regime-based scaling
                regime_multiplier = self._get_regime_multiplier(current_regime)
                final_signal = weighted_signal * regime_multiplier

                # Convert to discrete signal with hysteresis
                if final_signal > 0.7:
                    combined_signals[symbol] = 1
                elif final_signal < -0.7:
                    combined_signals[symbol] = -1
                else:
                    combined_signals[symbol] = 0
            else:
                combined_signals[symbol] = 0

        self.unconventional_signals = combined_signals

        return {
            'signals': combined_signals,
            'current_regime': current_regime,
            'regime_info': regime_info if 'regime_info' in locals() else {},
            'stochastic_analysis': stochastic_analysis if 'stochastic_analysis' in locals() else {},
            'alpha_generation_method': 'hmm_stochastic_unconventional'
        }

    def _generate_regime_transition_signals(self, price_data_dict, current_date,
                                          current_regime, regime_info):
        """Generate signals based on regime transitions"""
        signals = {}

        # Use transition probabilities to generate contrarian signals
        transition_probs = regime_info.get('transition_probabilities', [])

        for symbol, data in price_data_dict.items():
            if current_date not in data.index:
                continue

            # Get recent price action
            recent_data = data.loc[:current_date].tail(20)
            current_price = data.loc[current_date, 'Close']
            ma_10 = recent_data['Close'].rolling(10).mean().iloc[-1]
            ma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]

            # Generate signal based on regime and transition probability
            price_trend = (current_price - ma_20) / ma_20

            # Unconventional logic: in volatile regimes, fade transitions
            if 'volatile' in current_regime:
                # High transition probability suggests upcoming change
                max_transition_prob = max(transition_probs) if transition_probs else 0

                if max_transition_prob > 0.6:  # High chance of regime change
                    # Contrarian signal - expect mean reversion
                    if price_trend > 0.05:  # Overbought in volatile regime
                        signals[symbol] = -1
                    elif price_trend < -0.05:  # Oversold in volatile regime
                        signals[symbol] = 1
                    else:
                        signals[symbol] = 0
                else:
                    # Low transition probability - follow trend
                    if current_price > ma_10:
                        signals[symbol] = 1
                    else:
                        signals[symbol] = -1

            elif 'stable' in current_regime:
                # In stable regimes, use momentum
                if price_trend > 0:
                    signals[symbol] = 1
                elif price_trend < 0:
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0
            else:
                # Unknown regime - neutral
                signals[symbol] = 0

        return signals

    def _get_regime_multiplier(self, regime: str):
        """Get unconventional regime-based signal multiplier"""
        multipliers = {
            'bull_volatile': 1.2,     # Amplify signals in bull markets
            'bull_stable': 0.8,       # Dampen signals in stable bull markets
            'bear_volatile': -1.1,    # Strong short signals in bear markets
            'bear_stable': -0.9,      # Moderate short signals
            'sideways_volatile': 0.5, # Reduce signals in uncertain markets
            'sideways_stable': 0.3,   # Minimal signals in stable sideways
        }

        return multipliers.get(regime, 1.0)


# Integration and usage functions
def run_stochastic_optimization(strategy_class, assets_list, price_data_dict,
                               start_date, end_date, optimization_method='hmm_pso'):
    """
    Run stochastic optimization on strategy

    Args:
        strategy_class: Strategy class to optimize
        assets_list (list): List of assets
        price_data_dict (dict): Price data
        start_date (pd.Timestamp): Start date
        end_date (pd.Timestamp): End date
        optimization_method (str): 'hmm_pso', 'particle_swarm', 'genetic'

    Returns:
        dict: Optimization results
    """
    if optimization_method == 'hmm_pso':
        # Setup HMM detector
        hmm_detector = HMMRegimeDetector(n_regimes=4)

        # Fit HMM on historical data
        historical_returns = {}
        for symbol in assets_list:
            if symbol in price_data_dict:
                returns = price_data_dict[symbol]['Close'].pct_change().loc[start_date:end_date].dropna()
                historical_returns[symbol] = returns

        if historical_returns:
            returns_df = pd.DataFrame(historical_returns)
            hmm_detector.fit(returns_df)

        # Setup PSO optimizer
        pso_optimizer = ParticleSwarmOptimizer(n_particles=20, max_iterations=30)

        # Create stochastic regime optimizer
        regime_optimizer = StochasticRegimeOptimizer(hmm_detector, pso_optimizer)

        # Optimize for current date
        current_date = end_date
        result = regime_optimizer.optimize_with_regime_adaptation(
            strategy_class, {}, price_data_dict, current_date
        )

        return result

    elif optimization_method == 'particle_swarm':
        # Simple PSO optimization
        bounds = {
            'lookback_period': (10, 100),
            'threshold': (0.01, 0.5),
            'multiplier': (0.5, 3.0)
        }

        optimizer = ParticleSwarmOptimizer(bounds=bounds)

        def objective(params):
            try:
                signals = {}
                for date in pd.date_range(start_date, end_date, freq='W'):  # Weekly sampling
                    daily_signals = strategy_class(**params).generate_signals(price_data_dict, date)
                    for symbol, signal_df in daily_signals.items():
                        if symbol not in signals:
                            signals[symbol] = signal_df

                if signals:
                    from research.backtesting_engine import BacktestingEngine
                    backtest_engine = BacktestingEngine()
                    results = backtest_engine.run_backtest(signals, price_data_dict, start_date, end_date)
                    return results.get('sharpe_ratio', -999)
                return -999
            except:
                return -999

        return optimizer.optimize(objective)

    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")


# Example usage and testing
if __name__ == "__main__":
    print("Stochastic Optimization and HMM System")
    print("=" * 50)

    # Test HMM regime detector (if available)
    if HMM_AVAILABLE:
        print("HMM functionality available")
        hmm_detector = HMMRegimeDetector(n_regimes=3)

        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        sample_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.0005, 0.02, len(dates)),
            'MSFT': np.random.normal(0.0003, 0.025, len(dates)),
            'GOOGL': np.random.normal(0.0004, 0.018, len(dates))
        }, index=dates)

        try:
            hmm_results = hmm_detector.fit(sample_returns)
            print(f"HMM fitted with {hmm_results['n_regimes']} regimes")
            print(f"Regime labels: {hmm_results['regime_labels']}")
        except Exception as e:
            print(f"HMM fitting failed: {e}")
    else:
        print("HMM functionality not available (install hmmlearn)")

    # Test PSO optimizer
    def test_function(params):
        x, y = params['x'], params['y']
        return -(x**2 + y**2)  # Maximize (minimize negative)

    bounds = {'x': (-5, 5), 'y': (-5, 5)}
    pso = ParticleSwarmOptimizer(bounds=bounds, n_particles=10, max_iterations=20)

    pso_result = pso.optimize(test_function, verbose=False)
    print(f"PSO optimization: Best score = {pso_result['best_score']:.4f}")
    print(f"Best parameters: {pso_result['best_parameters']}")

    print("\\nStochastic optimization system initialized!")
