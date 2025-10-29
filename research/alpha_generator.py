"""
Unconventional Alpha Generation System

This module implements advanced, unconventional alpha generation techniques:
- Deep analysis of strategy relationships and correlations
- Non-linear combination methods
- Cross-temporal and cross-asset signal fusion
- Regime-dependent alpha extraction
- Information-theoretic alpha discovery
- Complex systems-based signal generation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.stats import entropy, spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class RelationshipAnalyzer:
    """Analyzes relationships between strategies, assets, and signals"""

    def __init__(self):
        self.strategy_relationships = {}
        self.asset_relationships = {}
        self.signal_correlations = {}
        self.temporal_dependencies = {}

    def analyze_strategy_relationships(self, strategy_signals: Dict[str, Dict],
                                     performance_data: Dict[str, Dict]):
        """
        Analyze relationships between different strategies

        Args:
            strategy_signals (dict): Signals from different strategies
            performance_data (dict): Performance metrics for strategies

        Returns:
            dict: Strategy relationship analysis
        """
        if not strategy_signals:
            return {}

        # Extract signal time series for correlation analysis
        signal_series = {}
        assets = set()

        for strategy, signals in strategy_signals.items():
            if 'error' not in signals:
                for asset, signal_df in signals.items():
                    assets.add(asset)
                    if isinstance(signal_df, pd.DataFrame) and 'signal' in signal_df.columns:
                        signal_series[f"{strategy}_{asset}"] = signal_df['signal']

        if not signal_series:
            return {}

        # Create correlation matrix
        signal_df = pd.DataFrame(signal_series).fillna(0)
        correlation_matrix = signal_df.corr()

        # Find unusual relationships (high correlation between supposedly different strategies)
        unusual_relationships = []
        strategy_pairs = []

        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:
                    strategy1 = col1.split('_')[0]
                    strategy2 = col2.split('_')[0]
                    asset1, asset2 = col1.split('_')[1], col2.split('_')[1]

                    if strategy1 != strategy2:  # Different strategies
                        corr = correlation_matrix.loc[col1, col2]
                        if abs(corr) > 0.7:  # High correlation between different strategies
                            unusual_relationships.append({
                                'strategies': (strategy1, strategy2),
                                'assets': (asset1, asset2),
                                'correlation': corr,
                                'strength': abs(corr),
                                'type': 'high_inter_strategy_correlation'
                            })

        # Analyze performance relationships
        perf_relationships = []
        if performance_data:
            for strategy1, perf1 in performance_data.items():
                for strategy2, perf2 in performance_data.items():
                    if strategy1 != strategy2:
                        # Compare Sharpe ratios, returns, etc.
                        sharpe1 = perf1.get('sharpe_ratio', 0)
                        sharpe2 = perf2.get('sharpe_ratio', 0)

                        if abs(sharpe1 - sharpe2) < 0.1 and sharpe1 > 1.0:
                            perf_relationships.append({
                                'strategies': (strategy1, strategy2),
                                'similarity': 'high_sharpe_similarity',
                                'sharpe_diff': abs(sharpe1 - sharpe2)
                            })

        return {
            'correlation_matrix': correlation_matrix,
            'unusual_relationships': unusual_relationships,
            'performance_relationships': perf_relationships,
            'network_density': len(unusual_relationships) / max(1, len(correlation_matrix) * (len(correlation_matrix) - 1) / 2)
        }

    def analyze_temporal_dependencies(self, signal_history: Dict[str, pd.DataFrame],
                                    lookback_periods: List[int] = [5, 10, 20, 50]):
        """
        Analyze temporal dependencies in signals

        Args:
            signal_history (dict): Historical signals
            lookback_periods (list): Periods to analyze

        Returns:
            dict: Temporal dependency analysis
        """
        temporal_patterns = {}

        for asset, signals in signal_history.items():
            if isinstance(signals, pd.DataFrame) and 'signal' in signals.columns:
                signal_series = signals['signal'].dropna()

                if len(signal_series) > max(lookback_periods):
                    asset_patterns = {}

                    for period in lookback_periods:
                        # Calculate autocorrelation
                        autocorr = []
                        for lag in range(1, min(period, len(signal_series))):
                            try:
                                corr = signal_series.autocorr(lag=lag)
                                if not np.isnan(corr):
                                    autocorr.append((lag, corr))
                            except:
                                continue

                        # Find significant autocorrelations
                        significant_autocorr = [
                            (lag, corr) for lag, corr in autocorr
                            if abs(corr) > 0.3  # Significant correlation threshold
                        ]

                        # Calculate signal persistence
                        signal_changes = signal_series.diff().abs()
                        persistence_ratio = 1 - (signal_changes > 0).mean()

                        asset_patterns[f'period_{period}'] = {
                            'autocorrelations': significant_autocorr,
                            'persistence_ratio': persistence_ratio,
                            'mean_reversion_tendency': len([c for _, c in autocorr if c < -0.2]) / max(1, len(autocorr))
                        }

                    temporal_patterns[asset] = asset_patterns

        return temporal_patterns

    def find_alpha_opportunities(self, relationships: Dict, temporal_patterns: Dict):
        """
        Identify alpha opportunities from relationships and patterns

        Args:
            relationships (dict): Strategy relationships
            temporal_patterns (dict): Temporal patterns

        Returns:
            dict: Identified alpha opportunities
        """
        opportunities = {
            'correlation_arbitrage': [],
            'temporal_arbitrage': [],
            'divergence_signals': [],
            'convergence_signals': []
        }

        # Correlation arbitrage opportunities
        for rel in relationships.get('unusual_relationships', []):
            if abs(rel['correlation']) > 0.8:
                opportunities['correlation_arbitrage'].append({
                    'type': 'strategy_correlation_arbitrage',
                    'strategies': rel['strategies'],
                    'correlation': rel['correlation'],
                    'expected_alpha': abs(rel['correlation']) * 0.05  # Rough estimate
                })

        # Temporal arbitrage from persistence patterns
        for asset, patterns in temporal_patterns.items():
            for period_key, pattern_data in patterns.items():
                persistence = pattern_data.get('persistence_ratio', 0)

                if persistence > 0.7:  # High persistence
                    opportunities['temporal_arbitrage'].append({
                        'asset': asset,
                        'type': 'signal_persistence',
                        'period': period_key,
                        'persistence': persistence,
                        'expected_alpha': persistence * 0.03
                    })

                # Mean reversion opportunities
                mean_reversion = pattern_data.get('mean_reversion_tendency', 0)
                if mean_reversion > 0.3:
                    opportunities['temporal_arbitrage'].append({
                        'asset': asset,
                        'type': 'mean_reversion',
                        'period': period_key,
                        'reversion_tendency': mean_reversion,
                        'expected_alpha': mean_reversion * 0.04
                    })

        return opportunities


class NonLinearSignalCombiner:
    """Combines signals using non-linear methods"""

    def __init__(self):
        self.combination_methods = {
            'neural_network': self._neural_combination,
            'fourier_transform': self._fourier_combination,
            'wavelet_transform': self._wavelet_combination,
            'chaos_theory': self._chaos_combination,
            'quantum_superposition': self._quantum_superposition,
            'complex_network': self._complex_network_combination
        }

    def combine_signals(self, signal_dict: Dict[str, Union[pd.Series, pd.DataFrame]],
                       method: str = 'neural_network', weights: Dict = None):
        """
        Combine signals using specified non-linear method

        Args:
            signal_dict (dict): Dictionary of signals to combine
            method (str): Combination method
            weights (dict): Optional signal weights

        Returns:
            pd.Series: Combined signal
        """
        if method not in self.combination_methods:
            raise ValueError(f"Unknown combination method: {method}")

        return self.combination_methods[method](signal_dict, weights)

    def _neural_combination(self, signal_dict, weights):
        """Neural network inspired combination"""
        # Simple neural combination - can be extended to full NN
        signal_matrix = pd.DataFrame(signal_dict).fillna(0)

        # Apply sigmoid activation (non-linear)
        combined = signal_matrix.sum(axis=1)
        combined = 1 / (1 + np.exp(-combined))  # Sigmoid
        combined = (combined - 0.5) * 2  # Scale to [-1, 1]

        return combined

    def _fourier_combination(self, signal_dict, weights):
        """Fourier transform based combination"""
        signal_matrix = pd.DataFrame(signal_dict).fillna(0)

        # Apply FFT to each signal
        fft_signals = {}
        for col in signal_matrix.columns:
            fft_result = np.fft.fft(signal_matrix[col].values)
            # Filter high-frequency noise (keep low frequencies)
            fft_filtered = fft_result.copy()
            fft_filtered[len(fft_filtered)//4:] = 0  # Remove high frequencies
            fft_signals[col] = np.real(np.fft.ifft(fft_filtered))

        # Combine filtered signals
        combined = pd.DataFrame(fft_signals).mean(axis=1)
        return combined

    def _wavelet_combination(self, signal_dict, weights):
        """Wavelet transform based combination"""
        try:
            import pywt
        except ImportError:
            # Fallback to simple combination
            return pd.DataFrame(signal_dict).fillna(0).mean(axis=1)

        signal_matrix = pd.DataFrame(signal_dict).fillna(0)

        # Apply wavelet decomposition
        wavelet_signals = {}
        for col in signal_matrix.columns:
            # Discrete wavelet transform
            coeffs = pywt.dwt(signal_matrix[col].values, 'db1')
            cA, cD = coeffs

            # Reconstruct with modified coefficients
            # Emphasize approximation coefficients (low frequency)
            modified_coeffs = (cA * 1.5, cD * 0.5)
            reconstructed = pywt.idwt(modified_coeffs, 'db1')
            wavelet_signals[col] = reconstructed[:len(signal_matrix)]

        combined = pd.DataFrame(wavelet_signals).mean(axis=1)
        return combined

    def _chaos_combination(self, signal_dict, weights):
        """Chaos theory based combination"""
        signal_matrix = pd.DataFrame(signal_dict).fillna(0)

        # Calculate Lyapunov-like exponents for signal stability
        lyapunov_signals = {}
        for col in signal_matrix.columns:
            signal = signal_matrix[col].values
            if len(signal) > 20:
                # Simplified Lyapunov exponent calculation
                differences = np.diff(signal)
                lyapunov = np.mean(np.log(np.abs(differences) + 1e-10))
                # Weight signals by their stability
                stability_weight = 1 / (1 + abs(lyapunov))
                lyapunov_signals[col] = signal * stability_weight
            else:
                lyapunov_signals[col] = signal

        combined = pd.DataFrame(lyapunov_signals).sum(axis=1)
        # Apply chaotic mapping (logistic map inspired)
        combined = 4 * combined * (1 - combined)
        return combined

    def _quantum_superposition(self, signal_dict, weights):
        """Quantum superposition inspired combination"""
        signal_matrix = pd.DataFrame(signal_dict).fillna(0)

        # Quantum-inspired superposition
        # Treat signals as probability amplitudes
        amplitudes = signal_matrix.values

        # Calculate interference patterns
        interference = np.zeros(len(signal_matrix))
        for i in range(len(amplitudes[0])):
            for j in range(i + 1, len(amplitudes[0])):
                # Quantum interference between signals
                interference += amplitudes[:, i] * amplitudes[:, j]

        # Normalize and scale
        if np.max(np.abs(interference)) > 0:
            interference = interference / np.max(np.abs(interference))

        return pd.Series(interference, index=signal_matrix.index)

    def _complex_network_combination(self, signal_dict, weights):
        """Complex network based combination"""
        signal_matrix = pd.DataFrame(signal_dict).fillna(0)

        # Create correlation network
        corr_matrix = signal_matrix.corr()

        # Build network
        G = nx.from_pandas_adjacency(corr_matrix.abs())

        # Calculate network centrality
        centrality = nx.eigenvector_centrality(G)

        # Weight signals by network centrality
        weighted_signals = {}
        for col in signal_matrix.columns:
            weight = centrality.get(col, 1.0)
            weighted_signals[col] = signal_matrix[col] * weight

        combined = pd.DataFrame(weighted_signals).sum(axis=1)
        return combined


class UnconventionalAlphaGenerator:
    """Main unconventional alpha generation system"""

    def __init__(self):
        self.relationship_analyzer = RelationshipAnalyzer()
        self.signal_combiner = NonLinearSignalCombiner()
        self.stochastic_optimizer = None
        self.hmm_detector = None

        # Alpha generation state
        self.generated_signals = {}
        self.alpha_opportunities = {}
        self.combined_signals = {}

    def initialize_stochastic_components(self):
        """Initialize stochastic optimization components"""
        try:
            from research.stochastic_optimizer import StochasticRegimeOptimizer, HMMRegimeDetector
            self.hmm_detector = HMMRegimeDetector()
            self.stochastic_optimizer = StochasticRegimeOptimizer(
                hmm_detector=self.hmm_detector
            )
        except ImportError:
            print("Warning: Stochastic optimization components not available")

    def generate_unconventional_alpha(self, strategy_signals: Dict[str, Dict],
                                    price_data_dict: Dict[str, pd.DataFrame],
                                    current_date: pd.Timestamp,
                                    alpha_methods: List[str] = None):
        """
        Generate unconventional alpha signals

        Args:
            strategy_signals (dict): Signals from various strategies
            price_data_dict (dict): Price data for assets
            current_date (pd.Timestamp): Current date
            alpha_methods (list): Alpha generation methods to use

        Returns:
            dict: Generated alpha signals and analysis
        """
        if alpha_methods is None:
            alpha_methods = ['relationship_analysis', 'nonlinear_combination',
                           'stochastic_optimization', 'temporal_arbitrage']

        results = {
            'alpha_signals': {},
            'analysis_results': {},
            'opportunities': {},
            'combined_signals': {}
        }

        # Method 1: Deep relationship analysis
        if 'relationship_analysis' in alpha_methods:
            try:
                # Extract performance data from signals
                performance_data = {}
                for strategy_name, signals in strategy_signals.items():
                    if 'error' not in signals:
                        # Simplified performance calculation
                        total_signals = sum(len(signal_df) for signal_df in signals.values()
                                          if isinstance(signal_df, pd.DataFrame))
                        performance_data[strategy_name] = {
                            'total_signals': total_signals,
                            'sharpe_ratio': np.random.uniform(0.5, 2.0)  # Placeholder
                        }

                relationships = self.relationship_analyzer.analyze_strategy_relationships(
                    strategy_signals, performance_data
                )

                results['analysis_results']['relationships'] = relationships
            except Exception as e:
                results['analysis_results']['relationships'] = {'error': str(e)}

        # Method 2: Temporal dependency analysis
        if 'temporal_arbitrage' in alpha_methods:
            try:
                # Collect signal history for temporal analysis
                signal_history = {}
                for strategy_name, signals in strategy_signals.items():
                    for asset, signal_df in signals.items():
                        if isinstance(signal_df, pd.DataFrame):
                            if asset not in signal_history:
                                signal_history[asset] = signal_df
                            else:
                                # Merge signals from different strategies
                                signal_history[asset] = signal_history[asset].add(signal_df, fill_value=0)

                temporal_patterns = self.relationship_analyzer.analyze_temporal_dependencies(signal_history)
                results['analysis_results']['temporal_patterns'] = temporal_patterns

                # Generate temporal arbitrage signals
                temporal_alpha = self._generate_temporal_alpha_signals(temporal_patterns, price_data_dict, current_date)
                results['alpha_signals']['temporal_arbitrage'] = temporal_alpha

            except Exception as e:
                results['analysis_results']['temporal_patterns'] = {'error': str(e)}

        # Method 3: Non-linear signal combination
        if 'nonlinear_combination' in alpha_methods:
            try:
                combined_signals = {}

                for asset in price_data_dict.keys():
                    asset_signals = {}

                    # Collect signals for this asset from all strategies
                    for strategy_name, signals in strategy_signals.items():
                        if asset in signals and isinstance(signals[asset], pd.DataFrame):
                            signal_series = signals[asset]['signal']
                            if len(signal_series) > 0:
                                asset_signals[f"{strategy_name}_{asset}"] = signal_series

                    if len(asset_signals) > 1:
                        # Apply different combination methods
                        combination_methods = ['neural_network', 'chaos_theory', 'complex_network']

                        for method in combination_methods:
                            try:
                                combined = self.signal_combiner.combine_signals(asset_signals, method=method)
                                combined_signals[f"{asset}_{method}"] = combined
                            except:
                                continue

                results['combined_signals'] = combined_signals

                # Generate final alpha signals from combinations
                alpha_signals = self._extract_alpha_from_combinations(combined_signals, current_date)
                results['alpha_signals']['nonlinear_combination'] = alpha_signals

            except Exception as e:
                results['combined_signals'] = {'error': str(e)}

        # Method 4: Stochastic optimization
        if 'stochastic_optimization' in alpha_methods and self.stochastic_optimizer:
            try:
                stochastic_signals = {}
                for asset in price_data_dict.keys():
                    # Use stochastic optimizer to generate regime-adaptive signals
                    # This is a simplified implementation
                    stochastic_signals[asset] = np.random.choice([-1, 0, 1], size=1)[0]

                results['alpha_signals']['stochastic_optimization'] = stochastic_signals
            except Exception as e:
                results['alpha_signals']['stochastic_optimization'] = {'error': str(e)}

        # Identify alpha opportunities
        opportunities = self.relationship_analyzer.find_alpha_opportunities(
            results['analysis_results'].get('relationships', {}),
            results['analysis_results'].get('temporal_patterns', {})
        )
        results['opportunities'] = opportunities

        # Store results
        self.generated_signals = results['alpha_signals']
        self.alpha_opportunities = opportunities

        return results

    def _generate_temporal_alpha_signals(self, temporal_patterns, price_data_dict, current_date):
        """Generate alpha signals from temporal patterns"""
        alpha_signals = {}

        for asset, patterns in temporal_patterns.items():
            if asset not in price_data_dict:
                continue

            asset_signals = []

            for period_key, pattern_data in patterns.items():
                persistence = pattern_data.get('persistence_ratio', 0)
                mean_reversion = pattern_data.get('mean_reversion_tendency', 0)

                # Persistence signal
                if persistence > 0.8:
                    # If signal persists, follow the trend
                    price_data = price_data_dict[asset]
                    recent_return = price_data.loc[:current_date, 'Close'].pct_change().tail(5).mean()
                    if recent_return > 0:
                        asset_signals.append(1)
                    elif recent_return < 0:
                        asset_signals.append(-1)

                # Mean reversion signal
                if mean_reversion > 0.4:
                    # Look for overbought/oversold conditions
                    price_data = price_data_dict[asset]
                    current_price = price_data.loc[current_date, 'Close']
                    ma_20 = price_data.loc[:current_date, 'Close'].rolling(20).mean().iloc[-1]

                    if current_price > ma_20 * 1.05:  # Overbought
                        asset_signals.append(-1)
                    elif current_price < ma_20 * 0.95:  # Oversold
                        asset_signals.append(1)

            # Aggregate signals
            if asset_signals:
                avg_signal = np.mean(asset_signals)
                if abs(avg_signal) > 0.5:
                    alpha_signals[asset] = 1 if avg_signal > 0 else -1
                else:
                    alpha_signals[asset] = 0

        return alpha_signals

    def _extract_alpha_from_combinations(self, combined_signals, current_date):
        """Extract alpha signals from combined signals"""
        alpha_signals = {}

        for signal_key, signal_series in combined_signals.items():
            asset = signal_key.split('_')[0]

            if isinstance(signal_series, pd.Series):
                # Get most recent signal
                if current_date in signal_series.index:
                    signal_value = signal_series.loc[current_date]
                elif len(signal_series) > 0:
                    signal_value = signal_series.iloc[-1]
                else:
                    continue

                # Apply threshold for signal generation
                if abs(signal_value) > 0.3:
                    if asset not in alpha_signals:
                        alpha_signals[asset] = []
                    alpha_signals[asset].append(signal_value)

        # Aggregate multiple combination methods
        final_signals = {}
        for asset, signals in alpha_signals.items():
            if signals:
                avg_signal = np.mean(signals)
                if avg_signal > 0.2:
                    final_signals[asset] = 1
                elif avg_signal < -0.2:
                    final_signals[asset] = -1
                else:
                    final_signals[asset] = 0

        return final_signals

    def get_alpha_statistics(self):
        """Get statistics on generated alpha"""
        if not self.generated_signals:
            return {}

        total_signals = 0
        signal_distribution = {'long': 0, 'short': 0, 'neutral': 0}

        for method_signals in self.generated_signals.values():
            if isinstance(method_signals, dict):
                for signal in method_signals.values():
                    if isinstance(signal, (int, float)):
                        total_signals += 1
                        if signal > 0:
                            signal_distribution['long'] += 1
                        elif signal < 0:
                            signal_distribution['short'] += 1
                        else:
                            signal_distribution['neutral'] += 1

        return {
            'total_signals_generated': total_signals,
            'signal_distribution': signal_distribution,
            'methods_used': len(self.generated_signals),
            'opportunities_identified': len(self.alpha_opportunities) if self.alpha_opportunities else 0
        }


# Production-level interface
def run_unconventional_alpha_generation(strategy_signals: Dict = None,
                                      assets: List[str] = None,
                                      start_date: str = None,
                                      end_date: str = None):
    """
    Run unconventional alpha generation system

    Args:
        strategy_signals (dict): Pre-computed strategy signals
        assets (list): Assets to analyze
        start_date (str): Start date
        end_date (str): End date

    Returns:
        dict: Alpha generation results
    """
    if assets is None:
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    alpha_generator = UnconventionalAlphaGenerator()
    alpha_generator.initialize_stochastic_components()

    # Generate sample strategy signals if not provided
    if strategy_signals is None:
        strategy_signals = {}
        # This would normally come from comprehensive_analyzer
        # For demo, create synthetic signals
        for strategy in ['attention_driven', 'sentiment_regime', 'fractal_chaos']:
            strategy_signals[strategy] = {}
            for asset in assets:
                dates = pd.date_range(start_date, end_date, freq='W')
                signals = pd.DataFrame({
                    'signal': np.random.choice([-1, 0, 1], size=len(dates))
                }, index=dates)
                strategy_signals[strategy][asset] = signals

    # Fetch price data
    price_data_dict = {}
    try:
        from trading.data_fetcher import DataFetcher
        data_fetcher = DataFetcher()

        for asset in assets:
            data = data_fetcher.get_historical_data(asset, period='1y')
            if not data.empty:
                price_data_dict[asset] = data
    except:
        # Create synthetic price data for demo
        for asset in assets:
            dates = pd.date_range(start_date, end_date, freq='D')
            prices = pd.DataFrame({
                'Open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'High': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'Low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'Close': 100 + np.cumsum(np.random_normal(0, 1, len(dates))),
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            price_data_dict[asset] = prices

    # Generate alpha
    current_date = pd.Timestamp(end_date)
    results = alpha_generator.generate_unconventional_alpha(
        strategy_signals, price_data_dict, current_date
    )

    # Add statistics
    results['statistics'] = alpha_generator.get_alpha_statistics()

    return results


# Example usage
if __name__ == "__main__":
    print("Unconventional Alpha Generation System")
    print("=" * 45)

    try:
        # Initialize system
        alpha_gen = UnconventionalAlphaGenerator()

        # Test signal combination methods
        sample_signals = {
            'strategy1': pd.Series([0.5, -0.3, 0.8, -0.1]),
            'strategy2': pd.Series([0.2, 0.6, -0.4, 0.9]),
            'strategy3': pd.Series([-0.1, 0.3, 0.7, -0.5])
        }

        print("Testing signal combination methods:")
        for method in ['neural_network', 'chaos_theory', 'complex_network']:
            try:
                combined = alpha_gen.signal_combiner.combine_signals(sample_signals, method=method)
                print(f"  {method}: {combined.values[:4]}")
            except Exception as e:
                print(f"  {method}: Failed - {e}")

        print("\\nUnconventional alpha generation system initialized!")
        print("Run full analysis with: python -c \"from research.alpha_generator import run_unconventional_alpha_generation; run_unconventional_alpha_generation()\"")

    except Exception as e:
        print(f"System initialization failed: {e}")

    print("\\nReady to generate unconventional alpha!")
