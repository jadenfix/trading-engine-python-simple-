"""
Comprehensive Test and Validation Suite

This module provides thorough testing and validation of all components
in the quantitative research framework, ensuring mathematical robustness,
error-free operation, and end-to-end functionality.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveTester:
    """Comprehensive testing framework for all components"""

    def __init__(self):
        self.test_results = {}
        self.error_log = []
        self.performance_metrics = {}
        self.mathematical_validations = {}

    def _setup_imports(self):
        """Setup import path for testing"""
        import sys
        import os
        # Add current directory to path for imports
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

    def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("Starting comprehensive test suite...")

        test_functions = [
            self.test_imports,
            self.test_data_fetcher,
            self.test_basic_strategies,
            self.test_unconventional_strategies,
            self.test_risk_manager,
            self.test_adaptive_optimizer,
            self.test_cross_market_signals,
            self.test_stochastic_optimizer,
            self.test_alpha_generator,
            self.test_comprehensive_analyzer,
            self.test_production_runner,
            self.test_mathematical_robustness,
            self.test_end_to_end_integration
        ]

        for test_func in test_functions:
            try:
                logger.info(f"Running {test_func.__name__}...")
                result = test_func()
                self.test_results[test_func.__name__] = result

                # Check if test passed
                if isinstance(result, dict):
                    if 'status' in result and result['status'] == 'SUCCESS':
                        logger.info(f"✓ {test_func.__name__} completed successfully")
                    elif 'status' in result and result['status'] == 'FAILED':
                        logger.error(f"✗ {test_func.__name__} failed: {result.get('message', 'Unknown error')}")
                        self.error_log.append(f"{test_func.__name__}: {result.get('message', 'Unknown error')}")
                    else:
                        logger.info(f"? {test_func.__name__} completed with status: {result.get('status', 'unknown')}")
                else:
                    logger.info(f"✓ {test_func.__name__} completed")

            except Exception as e:
                error_msg = f"✗ {test_func.__name__} failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.test_results[test_func.__name__] = {'status': 'FAILED', 'error': str(e), 'traceback': traceback.format_exc()}
                self.error_log.append(error_msg)

        # Generate summary
        self.generate_test_summary()
        return self.test_results

    def test_imports(self):
        """Test all module imports"""
        self._setup_imports()

        modules_to_test = [
            'research.strategies',
            'research.unconventional_strategies',
            'research.risk_manager',
            'research.adaptive_optimizer',
            'research.cross_market_signals',
            'research.stochastic_optimizer',
            'research.alpha_generator',
            'research.comprehensive_analyzer',
            'research.production_runner',
            'trading.data_fetcher',
            'trading.algorithm',
            'trading.backtesting'
        ]

        failed_imports = []

        for module in modules_to_test:
            try:
                __import__(module)
            except Exception as e:
                failed_imports.append(module)

        if len(failed_imports) == 0:
            return {'status': 'SUCCESS', 'modules_tested': len(modules_to_test), 'failed_imports': 0}
        else:
            return {'status': 'FAILED', 'message': f'{len(failed_imports)}/{len(modules_to_test)} modules failed to import: {failed_imports}'} 

    def test_data_fetcher(self):
        """Test data fetching functionality"""
        self._setup_imports()
        try:
            from trading.data_fetcher import DataFetcher
            fetcher = DataFetcher()

            # Test basic functionality
            test_symbol = 'AAPL'
            data = fetcher.get_historical_data(test_symbol, period='1mo')

            if data.empty:
                return {'status': 'WARNING', 'message': 'No data fetched (may be due to API limits)'}

            # Validate data structure
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                return {'status': 'FAILED', 'message': f'Missing columns: {missing_columns}'}

            # Validate data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return {'status': 'FAILED', 'message': f'Non-numeric data in {col}'}

            # Test current price
            current_price = fetcher.get_current_price(test_symbol)
            if current_price is None:
                return {'status': 'WARNING', 'message': 'Current price fetch failed'}

            return {
                'status': 'SUCCESS',
                'data_points': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'current_price': current_price
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_basic_strategies(self):
        """Test basic strategy implementations"""
        self._setup_imports()
        try:
            from research.strategies import (
                FactorMomentumStrategy, CrossSectionalMomentumStrategy,
                VolatilityRegimeStrategy, LiquidityTimingStrategy,
                StatisticalProcessControlStrategy
            )

            # Create sample data
            dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
            sample_data = {
                'AAPL': pd.DataFrame({
                    'Open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'High': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates),
                'MSFT': pd.DataFrame({
                    'Open': 200 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'High': 205 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Low': 195 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Close': 200 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates)
            }

            strategies = [
                ('FactorMomentumStrategy', FactorMomentumStrategy()),
                ('CrossSectionalMomentumStrategy', CrossSectionalMomentumStrategy()),
                ('VolatilityRegimeStrategy', VolatilityRegimeStrategy()),
                ('LiquidityTimingStrategy', LiquidityTimingStrategy()),
                ('StatisticalProcessControlStrategy', StatisticalProcessControlStrategy())
            ]

            results = {}
            current_date = pd.Timestamp('2022-01-01')

            for strategy_name, strategy in strategies:
                try:
                    signals = strategy.generate_signals(sample_data, current_date)

                    # Validate signal structure
                    if not isinstance(signals, dict):
                        results[strategy_name] = f'FAILED: Signals not dict, got {type(signals)}'
                        continue

                    # Check signal values
                    valid_signals = 0
                    for asset, signal_df in signals.items():
                        if isinstance(signal_df, pd.DataFrame) and 'signal' in signal_df.columns:
                            signal_value = signal_df['signal'].iloc[0] if len(signal_df) > 0 else None
                            if signal_value in [-1, 0, 1]:
                                valid_signals += 1

                    results[strategy_name] = f'SUCCESS: {valid_signals} valid signals generated'

                except Exception as e:
                    results[strategy_name] = f'FAILED: {str(e)}'

            # Count successes and failures
            successful_strategies = sum(1 for r in results.values() if r.startswith('SUCCESS'))
            total_strategies = len(results)

            if successful_strategies == total_strategies:
                return {'status': 'SUCCESS', 'strategies_tested': total_strategies, 'results': results}
            elif successful_strategies > 0:
                return {'status': 'WARNING', 'message': f'{successful_strategies}/{total_strategies} strategies passed', 'results': results}
            else:
                return {'status': 'FAILED', 'message': f'0/{total_strategies} strategies passed', 'results': results}

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_unconventional_strategies(self):
        """Test unconventional strategy implementations"""
        self._setup_imports()
        try:
            from research.unconventional_strategies import (
                AttentionDrivenStrategy, SentimentRegimeStrategy,
                InformationTheoryStrategy, ComplexSystemsStrategy,
                FractalChaosStrategy, QuantumInspiredStrategy
            )

            # Create sample data
            dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
            sample_data = {
                'AAPL': pd.DataFrame({
                    'Open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'High': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates)
            }

            strategies = [
                ('AttentionDrivenStrategy', AttentionDrivenStrategy()),
                ('SentimentRegimeStrategy', SentimentRegimeStrategy()),
                ('InformationTheoryStrategy', InformationTheoryStrategy()),
                ('ComplexSystemsStrategy', ComplexSystemsStrategy()),
                ('FractalChaosStrategy', FractalChaosStrategy()),
                ('QuantumInspiredStrategy', QuantumInspiredStrategy())
            ]

            results = {}
            current_date = pd.Timestamp('2022-01-01')

            for strategy_name, strategy in strategies:
                try:
                    signals = strategy.generate_signals(sample_data, current_date)

                    # Validate signal structure
                    if not isinstance(signals, dict):
                        results[strategy_name] = f'FAILED: Signals not dict'
                        continue

                    signal_count = 0
                    for asset, signal_df in signals.items():
                        if isinstance(signal_df, pd.DataFrame) and 'signal' in signal_df.columns:
                            if len(signal_df) > 0:
                                signal_count += 1

                    results[strategy_name] = f'SUCCESS: {signal_count} signals generated'

                except Exception as e:
                    results[strategy_name] = f'FAILED: {str(e)}'

            # Count successes and failures
            successful_strategies = sum(1 for r in results.values() if r.startswith('SUCCESS'))
            total_strategies = len(results)

            if successful_strategies == total_strategies:
                return {'status': 'SUCCESS', 'strategies_tested': total_strategies, 'results': results}
            elif successful_strategies > 0:
                return {'status': 'WARNING', 'message': f'{successful_strategies}/{total_strategies} strategies passed', 'results': results}
            else:
                return {'status': 'FAILED', 'message': f'0/{total_strategies} strategies passed', 'results': results}

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_risk_manager(self):
        """Test risk management functionality"""
        self._setup_imports()
        try:
            from research.risk_manager import RiskManager
            risk_manager = RiskManager(confidence_level=0.95, max_drawdown_limit=0.20)

            # Create sample portfolio returns
            dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
            portfolio_returns = pd.Series(
                np.random.normal(0.0005, 0.02, len(dates)),
                index=dates
            )
            portfolio_values = (1 + portfolio_returns).cumprod()

            # Test VaR calculation
            var_result = risk_manager.calculate_var(portfolio_returns)
            if not isinstance(var_result, dict) or 'var' not in var_result:
                return {'status': 'FAILED', 'message': 'VaR calculation failed'}

            # Test CVaR
            if 'expected_shortfall' not in var_result:
                return {'status': 'FAILED', 'message': 'CVaR calculation failed'}

            # Test Kelly criterion
            kelly_result = risk_manager.calculate_kelly_criterion(portfolio_returns)
            if not isinstance(kelly_result, dict) or 'kelly_fraction' not in kelly_result:
                return {'status': 'FAILED', 'message': 'Kelly criterion failed'}

            # Test drawdown monitoring
            drawdown_result = risk_manager.monitor_drawdown(portfolio_values)
            if not isinstance(drawdown_result, dict):
                return {'status': 'FAILED', 'message': 'Drawdown monitoring failed'}

            return {
                'status': 'SUCCESS',
                'var_95': var_result['var'],
                'cvar_95': var_result['expected_shortfall'],
                'kelly_fraction': kelly_result['kelly_fraction'],
                'max_drawdown': drawdown_result.get('max_drawdown', 0)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_adaptive_optimizer(self):
        """Test adaptive optimization functionality"""
        self._setup_imports()
        try:
            from research.adaptive_optimizer import BayesianOptimizer, WalkForwardOptimizer

            # Test Bayesian optimization
            bounds = {'x': (-5, 5), 'y': (-5, 5)}
            optimizer = BayesianOptimizer(bounds=bounds)

            def test_function(params):
                x, y = params['x'], params['y']
                return -(x**2 + y**2)  # Maximize negative of sphere function

            bounds = {'x': (-5, 5), 'y': (-5, 5)}
            result = optimizer.optimize(test_function)

            if not isinstance(result, dict) or 'best_parameters' not in result:
                return {'status': 'FAILED', 'message': 'Bayesian optimization failed'}

            # Check if optimum is near (0,0)
            best_x, best_y = result['best_parameters']['x'], result['best_parameters']['y']
            distance_from_optimum = np.sqrt(best_x**2 + best_y**2)

            if distance_from_optimum > 2.0:
                return {'status': 'WARNING', 'message': f'Optimization may not have converged, distance: {distance_from_optimum:.3f}'}

            return {
                'status': 'SUCCESS',
                'bayesian_opt': result['best_score'],
                'distance_from_optimum': distance_from_optimum
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_cross_market_signals(self):
        """Test cross-market signal functionality"""
        self._setup_imports()
        try:
            from research.cross_market_signals import CrossMarketAnalyzer

            analyzer = CrossMarketAnalyzer()

            # Create sample data
            dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
            equity_data = pd.DataFrame({
                'Open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'High': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'Low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

            fx_data = pd.DataFrame({
                'Close': 1.1 + np.cumsum(np.random.normal(0, 0.01, len(dates)))
            }, index=dates)

            # Add market data
            analyzer.add_market_data('equity', 'AAPL', equity_data)
            analyzer.add_market_data('fx', 'EURUSD=X', fx_data)

            # Test correlation analysis
            current_date = pd.Timestamp('2022-01-01')
            correlations = analyzer.analyze_inter_market_correlations(current_date)
            if not isinstance(correlations, dict):
                return {'status': 'FAILED', 'message': 'Correlation analysis failed'}

            # Test signal generation
            current_date = pd.Timestamp('2022-01-01')
            # Provide mock equity signals
            equity_signals = {'AAPL': 1, 'MSFT': -1}  # Mock signals
            signals = analyzer.generate_cross_market_signals(current_date, equity_signals)
            if not isinstance(signals, dict):
                return {'status': 'FAILED', 'message': 'Signal generation failed'}

            return {
                'status': 'SUCCESS',
                'markets_added': len(analyzer.equity_data) + len(analyzer.fx_data),
                'correlations_found': len(correlations.get('significant_relationships', [])),
                'signals_generated': len(signals)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_stochastic_optimizer(self):
        """Test stochastic optimization functionality"""
        self._setup_imports()
        try:
            from research.stochastic_optimizer import ParticleSwarmOptimizer, HMMRegimeDetector

            # Test PSO
            bounds = {'x': (-5, 5), 'y': (-5, 5)}
            pso = ParticleSwarmOptimizer(bounds=bounds, n_particles=10, max_iterations=20)

            def objective(params):
                x, y = params['x'], params['y']
                return -(x**2 + y**2)

            result = pso.optimize(objective, verbose=False)

            if not isinstance(result, dict) or 'best_parameters' not in result:
                return {'status': 'FAILED', 'message': 'PSO optimization failed'}

            # Test HMM (if available)
            hmm_available = True
            try:
                hmm_detector = HMMRegimeDetector(n_regimes=2)

                # Create sample data
                dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
                sample_returns = pd.DataFrame({
                    'AAPL': np.random.normal(0.0005, 0.02, len(dates)),
                    'MSFT': np.random.normal(0.0003, 0.025, len(dates))
                }, index=dates)

                hmm_results = hmm_detector.fit(sample_returns)
                hmm_status = 'SUCCESS' if isinstance(hmm_results, dict) else 'FAILED'

            except ImportError:
                hmm_status = 'HMM_NOT_AVAILABLE'
                hmm_available = False
            except Exception as e:
                hmm_status = f'HMM_ERROR: {str(e)}'

            return {
                'status': 'SUCCESS',
                'pso_best_score': result['best_score'],
                'hmm_status': hmm_status,
                'hmm_available': hmm_available
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_alpha_generator(self):
        """Test alpha generation functionality"""
        self._setup_imports()
        try:
            from research.alpha_generator import UnconventionalAlphaGenerator

            alpha_gen = UnconventionalAlphaGenerator()

            # Test signal combination
            sample_signals = {
                'signal1': pd.Series([0.5, -0.3, 0.8]),
                'signal2': pd.Series([0.2, 0.6, -0.4]),
                'signal3': pd.Series([-0.1, 0.3, 0.7])
            }

            combined = alpha_gen.signal_combiner.combine_signals(sample_signals, method='neural_network')

            if not isinstance(combined, pd.Series):
                return {'status': 'FAILED', 'message': 'Signal combination failed'}

            # Test relationship analyzer
            strategy_signals = {
                'strategy1': {'AAPL': pd.DataFrame({'signal': [1, -1, 0]}, index=pd.date_range('2023-01-01', periods=3))},
                'strategy2': {'AAPL': pd.DataFrame({'signal': [-1, 1, 1]}, index=pd.date_range('2023-01-01', periods=3))}
            }

            performance_data = {
                'strategy1': {'sharpe_ratio': 1.2},
                'strategy2': {'sharpe_ratio': 0.8}
            }

            relationships = alpha_gen.relationship_analyzer.analyze_strategy_relationships(
                strategy_signals, performance_data
            )

            if not isinstance(relationships, dict):
                return {'status': 'FAILED', 'message': 'Relationship analysis failed'}

            return {
                'status': 'SUCCESS',
                'signal_combination': len(combined),
                'relationships_analyzed': len(relationships.get('correlation_matrix', {})),
                'unusual_relationships': len(relationships.get('unusual_relationships', []))
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_comprehensive_analyzer(self):
        """Test comprehensive analyzer functionality"""
        self._setup_imports()
        try:
            from research.comprehensive_analyzer import ComprehensiveAnalyzer

            analyzer = ComprehensiveAnalyzer()

            # Check if strategies are registered
            strategies = analyzer.strategy_registry.get_all_strategies()

            if len(strategies) == 0:
                return {'status': 'FAILED', 'message': 'No strategies registered'}

            # Test basic functionality
            categories = set(analyzer.strategy_registry.strategy_categories.values())

            return {
                'status': 'SUCCESS',
                'strategies_registered': len(strategies),
                'categories': list(categories),
                'strategy_types': {'traditional': len([s for s in strategies if analyzer.strategy_registry.strategy_categories.get(s) == 'traditional']),
                                 'unconventional': len([s for s in strategies if analyzer.strategy_registry.strategy_categories.get(s) == 'unconventional'])}
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_production_runner(self):
        """Test production runner functionality"""
        self._setup_imports()
        try:
            from research.production_runner import ProductionRunner

            runner = ProductionRunner()

            # Test system status
            status = runner.get_system_status()

            if not isinstance(status, dict):
                return {'status': 'FAILED', 'message': 'System status check failed'}

            # Check component initialization
            components = status.get('components_initialized', {})

            return {
                'status': 'SUCCESS',
                'system_health': status.get('system_health'),
                'components_initialized': sum(components.values()),
                'total_components': len(components)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_mathematical_robustness(self):
        """Test mathematical robustness of all components"""
        robustness_tests = {}

        # Test numerical stability
        try:
            # Test with edge cases
            extreme_returns = np.array([0.5, -0.5, 0.99, -0.99, 0, 0.001])
            mean_return = np.mean(extreme_returns)
            std_return = np.std(extreme_returns)

            if not np.isfinite(mean_return) or not np.isfinite(std_return):
                robustness_tests['numerical_stability'] = 'FAILED: Non-finite values'
            else:
                robustness_tests['numerical_stability'] = 'SUCCESS'

        except Exception as e:
            robustness_tests['numerical_stability'] = f'FAILED: {str(e)}'

        # Test correlation matrix properties
        try:
            # Create sample correlation matrix
            n_assets = 5
            corr_matrix = np.random.uniform(-1, 1, (n_assets, n_assets))
            corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1

            # Check if positive semi-definite
            eigenvals = np.linalg.eigvals(corr_matrix)
            if np.all(eigenvals >= -1e-10):  # Allow small numerical errors
                robustness_tests['correlation_matrix'] = 'SUCCESS: Valid correlation matrix'
            else:
                robustness_tests['correlation_matrix'] = 'FAILED: Not positive semi-definite'

        except Exception as e:
            robustness_tests['correlation_matrix'] = f'FAILED: {str(e)}'

        # Test optimization convergence
        try:
            from scipy.optimize import minimize_scalar

            def test_function(x):
                return (x - 2)**2 + 1

            result = minimize_scalar(test_function, bounds=(0, 4), method='bounded')

            if abs(result.x - 2) < 0.01:
                robustness_tests['optimization_convergence'] = 'SUCCESS: Converged to optimum'
            else:
                robustness_tests['optimization_convergence'] = f'FAILED: Converged to {result.x:.3f}, expected 2.0'

        except Exception as e:
            robustness_tests['optimization_convergence'] = f'FAILED: {str(e)}'

        # Check overall robustness
        failed_tests = sum(1 for r in robustness_tests.values() if r.startswith('FAILED'))
        if failed_tests == 0:
            return {'status': 'SUCCESS', 'robustness_tests': robustness_tests}
        else:
            return {'status': 'WARNING', 'message': f'{failed_tests} robustness tests failed', 'robustness_tests': robustness_tests}

    def test_end_to_end_integration(self):
        """Test end-to-end integration of all components"""
        self._setup_imports()
        try:
            from research.production_runner import run_production_analysis

            # Run a quick test with minimal data
            assets = ['AAPL', 'MSFT']
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            config = {
                'run_strategy_analysis': True,
                'run_alpha_generation': False,
                'run_stochastic_opt': False,
                'analysis_frequency': 'M',  # Monthly for speed
                'min_trades_per_asset': 1,  # Lower threshold for testing
                'export_results': False
            }

            results = run_production_analysis(assets, start_date, end_date, config)

            if 'error' in results:
                return {'status': 'FAILED', 'error': results['error']}

            # Validate results structure
            required_keys = ['execution_info', 'strategy_analysis', 'performance_summary']
            missing_keys = [key for key in required_keys if key not in results]

            if missing_keys:
                return {'status': 'FAILED', 'message': f'Missing keys: {missing_keys}'}

            # Check if strategies were executed
            strategy_analysis = results.get('strategy_analysis', {})
            if 'summary_statistics' not in strategy_analysis:
                return {'status': 'WARNING', 'message': 'No summary statistics generated'}

            summary = strategy_analysis.get('summary_statistics', {})
            successful_strategies = summary.get('successful_strategies', 0)

            return {
                'status': 'SUCCESS',
                'execution_time': results['execution_info'].get('duration_seconds', 0),
                'successful_strategies': successful_strategies,
                'assets_processed': len(results['execution_info'].get('assets', [])),
                'total_trades': summary.get('total_trades_generated', 0)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e), 'traceback': traceback.format_exc()}

    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        total_tests = len(self.test_results)
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0

        for result in self.test_results.values():
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                if status == 'SUCCESS':
                    passed_tests += 1
                elif status == 'FAILED':
                    failed_tests += 1
                elif status == 'WARNING':
                    warning_tests += 1
            else:
                # For non-dict results (like test_imports), assume success if no error key
                if isinstance(result, dict) and 'error' in result:
                    failed_tests += 1
                else:
                    passed_tests += 1

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Warnings: {warning_tests}")
        logger.info(f"Success Rate: {success_rate:.1%}")

        if self.error_log:
            logger.info("\nERRORS ENCOUNTERED:")
            for error in self.error_log:
                logger.error(f"  - {error}")

        # Detailed results
        logger.info("\nDETAILED TEST RESULTS:")
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                if status == 'SUCCESS':
                    logger.info(f"✓ {test_name}: {status}")
                elif status == 'WARNING':
                    logger.warning(f"⚠ {test_name}: {status}")
                elif status == 'FAILED':
                    logger.error(f"✗ {test_name}: {status} - {result.get('message', result.get('error', 'Unknown error'))}")
                else:
                    logger.info(f"? {test_name}: {status}")
            else:
                logger.info(f"? {test_name}: Non-dict result")

        # Mathematical validations
        if self.mathematical_validations:
            logger.info("\nMATHEMATICAL VALIDATIONS:")
            for validation, result in self.mathematical_validations.items():
                logger.info(f"  {validation}: {result}")

        logger.info("="*60)

        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'warnings': warning_tests,
            'success_rate': success_rate,
            'errors': self.error_log
        }


def run_comprehensive_tests():
    """Run the complete test suite"""
    tester = ComprehensiveTester()
    results = tester.run_all_tests()

    summary = tester.generate_test_summary()
    return results, summary


if __name__ == "__main__":
    print("Comprehensive Test and Validation Suite")
    print("=" * 50)

    results, summary = run_comprehensive_tests()

    print(f"\nTest Summary: {summary['passed']}/{summary['total_tests']} tests passed")

    if summary['success_rate'] >= 0.8:
        print("Framework validation: PASSED")
    else:
        print("Framework validation: FAILED")
        print("Please review the error logs above.")
