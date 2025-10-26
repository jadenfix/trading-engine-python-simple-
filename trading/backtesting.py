"""
Advanced backtesting framework with validation and optimization capabilities
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import itertools


class AdvancedBacktester:
    """Advanced backtesting framework with validation and optimization"""

    def __init__(self, algorithm_class, data_fetcher):
        """
        Initialize advanced backtester

        Args:
            algorithm_class: TradingAlgorithm class
            data_fetcher: DataFetcher instance
        """
        self.algorithm_class = algorithm_class
        self.data_fetcher = data_fetcher

    def walk_forward_optimization(self, symbols: List[str], start_date: str, end_date: str,
                                 train_window: int = 252, test_window: int = 63,
                                 step_size: int = 21, strategy_params: Dict = None) -> Dict:
        """
        Perform walk-forward optimization to validate strategy robustness

        Args:
            symbols: List of symbols to test
            start_date: Start date for testing
            end_date: End date for testing
            train_window: Training window in days
            test_window: Testing window in days
            step_size: Step size for rolling window
            strategy_params: Strategy parameters to optimize

        Returns:
            Dict: Walk-forward optimization results
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        results = {
            'train_periods': [],
            'test_periods': [],
            'train_results': [],
            'test_results': [],
            'combined_results': [],
            'parameter_stability': {},
            'out_of_sample_performance': []
        }

        current_start = start

        while current_start + timedelta(days=train_window + test_window) <= end:
            train_end = current_start + timedelta(days=train_window)
            test_end = train_end + timedelta(days=test_window)

            # Training period
            train_results = self._run_backtest_period(symbols, current_start, train_end,
                                                    strategy_params, 'train')

            # Testing period (out-of-sample)
            test_results = self._run_backtest_period(symbols, train_end, test_end,
                                                   strategy_params, 'test')

            results['train_periods'].append((current_start, train_end))
            results['test_periods'].append((train_end, test_end))
            results['train_results'].append(train_results)
            results['test_results'].append(test_results)

            # Combined results
            if train_results and test_results:
                combined_return = (train_results['total_return'] * (train_window / (train_window + test_window)) +
                                 test_results['total_return'] * (test_window / (train_window + test_window)))
                results['combined_results'].append(combined_return)

            current_start += timedelta(days=step_size)

        # Calculate stability metrics
        results['parameter_stability'] = self._calculate_parameter_stability(results)
        results['out_of_sample_performance'] = self._analyze_out_of_sample_performance(results)

        return results

    def monte_carlo_simulation(self, symbols: List[str], start_date: str, end_date: str,
                              num_simulations: int = 1000, strategy_params: Dict = None) -> Dict:
        """
        Run Monte Carlo simulation to assess strategy robustness

        Args:
            symbols: List of symbols to test
            start_date: Start date for testing
            end_date: End date for testing
            num_simulations: Number of Monte Carlo simulations
            strategy_params: Strategy parameters

        Returns:
            Dict: Monte Carlo simulation results
        """
        # Get base data
        base_results = self._run_backtest_period(symbols, start_date, end_date,
                                               strategy_params, 'full')

        if not base_results:
            return {"error": "No base results available"}

        # Extract trade sequence
        trades = base_results.get('trades', pd.DataFrame())

        if trades.empty:
            return {"error": "No trades in base results"}

        # Run Monte Carlo simulations
        simulation_results = []

        for i in range(num_simulations):
            # Bootstrap trade sequence
            bootstrapped_trades = trades.sample(n=len(trades), replace=True).sort_values('date')

            # Simulate with bootstrapped trades
            sim_result = self._simulate_with_trades(symbols, bootstrapped_trades, strategy_params)
            simulation_results.append(sim_result)

        # Analyze simulation results
        returns = [r['total_return'] for r in simulation_results]
        drawdowns = [r['max_drawdown'] for r in simulation_results]

        return {
            'base_results': base_results,
            'simulation_results': simulation_results,
            'return_distribution': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95),
                'min': np.min(returns),
                'max': np.max(returns)
            },
            'drawdown_distribution': {
                'mean': np.mean(drawdowns),
                'std': np.std(drawdowns),
                'percentile_5': np.percentile(drawdowns, 5),
                'percentile_95': np.percentile(drawdowns, 95)
            },
            'probability_of_loss': len([r for r in returns if r < 0]) / len(returns),
            'best_case': max(returns),
            'worst_case': min(returns)
        }

    def parameter_sensitivity_analysis(self, symbols: List[str], start_date: str, end_date: str,
                                     parameter_ranges: Dict, num_combinations: int = 50) -> Dict:
        """
        Analyze parameter sensitivity across different combinations

        Args:
            symbols: List of symbols to test
            start_date: Start date for testing
            end_date: End date for testing
            parameter_ranges: Dictionary of parameter ranges to test
            num_combinations: Number of parameter combinations to test

        Returns:
            Dict: Parameter sensitivity analysis results
        """
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        # Create combinations (limit to avoid too many tests)
        if len(param_names) > 3:
            # For many parameters, use latin hypercube sampling
            combinations = self._latin_hypercube_sample(parameter_ranges, num_combinations)
        else:
            # For few parameters, test all combinations
            all_combinations = list(itertools.product(*param_values))
            step = max(1, len(all_combinations) // num_combinations)
            combinations = all_combinations[::step][:num_combinations]

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                result = self._run_backtest_period(symbols, start_date, end_date, params, 'full')
                if result:
                    results.append({
                        'parameters': params,
                        'total_return': result.get('total_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'total_trades': result.get('total_trades', 0),
                        'win_rate': result.get('win_rate', 0)
                    })
            except Exception as e:
                print(f"Warning: Parameter combination {params} failed: {str(e)}")
                continue

        # Analyze results
        if not results:
            return {"error": "No successful parameter combinations"}

        # Find best parameters for each metric
        best_by_return = max(results, key=lambda x: x['total_return'])
        best_by_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
        best_by_drawdown = min(results, key=lambda x: x['max_drawdown'])

        return {
            'all_results': results,
            'best_by_return': best_by_return,
            'best_by_sharpe': best_by_sharpe,
            'best_by_drawdown': best_by_drawdown,
            'parameter_importance': self._analyze_parameter_importance(results, param_names),
            'robustness_score': self._calculate_robustness_score(results)
        }

    def stress_testing(self, symbols: List[str], start_date: str, end_date: str,
                      stress_scenarios: List[Dict], strategy_params: Dict = None) -> Dict:
        """
        Run stress tests under various market conditions

        Args:
            symbols: List of symbols to test
            start_date: Start date for testing
            end_date: End date for testing
            stress_scenarios: List of stress test scenarios
            strategy_params: Strategy parameters

        Returns:
            Dict: Stress testing results
        """
        base_results = self._run_backtest_period(symbols, start_date, end_date,
                                               strategy_params, 'base')

        if not base_results:
            return {"error": "No base results available"}

        stress_results = {'base': base_results}

        for scenario in stress_scenarios:
            try:
                # Apply scenario modifications to data
                modified_data = self._apply_stress_scenario(symbols, start_date, end_date, scenario)

                # Run backtest with modified data
                scenario_result = self._run_backtest_with_data(modified_data, strategy_params)
                stress_results[scenario['name']] = scenario_result

            except Exception as e:
                print(f"Warning: Stress scenario {scenario['name']} failed: {str(e)}")
                stress_results[scenario['name']] = {"error": str(e)}

        return stress_results

    def _run_backtest_period(self, symbols: List[str], start_date, end_date,
                           strategy_params: Dict = None, period_name: str = 'full'):
        """Run backtest for a specific period"""
        try:
            algorithm = self.algorithm_class(
                initial_capital=100000,
                strategy=strategy_params.get('strategy', 'enhanced') if strategy_params else 'enhanced',
                risk_profile=strategy_params.get('risk_profile', 'medium') if strategy_params else 'medium'
            )

            # Override strategy parameters if provided
            if strategy_params:
                for param, value in strategy_params.items():
                    if hasattr(algorithm.strategy, param):
                        setattr(algorithm.strategy, param, value)

            # Get data for the period
            all_data = {}
            for symbol in symbols:
                data = self.data_fetcher.get_historical_data(symbol, start_date, end_date)
                if not data.empty:
                    all_data[symbol] = data

            if not all_data:
                return None

            # Find common date range
            common_start = max(data.index[0] for data in all_data.values())
            common_end = min(data.index[-1] for data in all_data.values())

            if common_start >= common_end:
                return None

            # Run backtest
            return algorithm.run_backtest(symbols, common_start, common_end)

        except Exception as e:
            print(f"Warning: Backtest failed for {period_name}: {str(e)}")
            return None

    def _simulate_with_trades(self, symbols: List[str], trades: pd.DataFrame,
                            strategy_params: Dict = None) -> Dict:
        """Simulate portfolio performance with given trades"""
        # Simplified simulation - in practice would need full portfolio simulation
        initial_capital = 100000

        if trades.empty:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

        # Calculate cumulative return from trades
        portfolio_value = initial_capital
        peak_value = initial_capital
        max_drawdown = 0
        total_profit = 0

        for _, trade in trades.iterrows():
            # Simplified trade simulation
            if trade['side'] == 'buy':
                # Assume position size based on capital
                position_value = portfolio_value * 0.1  # 10% of capital
                shares = position_value / trade['price']
                cost = shares * trade['price']
                portfolio_value -= cost
                total_profit -= cost
            else:  # sell
                # Find corresponding buy trade (simplified)
                revenue = shares * trade['price']
                portfolio_value += revenue
                total_profit += revenue - cost

        total_return = (portfolio_value - initial_capital) / initial_capital
        drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades)
        }

    def _latin_hypercube_sample(self, parameter_ranges: Dict, num_samples: int) -> List[Tuple]:
        """Generate Latin Hypercube samples for parameter combinations"""
        # Simplified LHS implementation
        samples = []

        for _ in range(num_samples):
            sample = []
            for param_name, param_range in parameter_ranges.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                    # Continuous parameter
                    value = np.random.uniform(param_range[0], param_range[1])
                else:
                    # Discrete parameter
                    value = np.random.choice(param_range)
                sample.append(value)
            samples.append(tuple(sample))

        return samples

    def _calculate_parameter_stability(self, results: Dict) -> Dict:
        """Calculate parameter stability across walk-forward periods"""
        if not results['combined_results']:
            return {}

        returns = results['combined_results']
        return {
            'return_volatility': np.std(returns),
            'return_consistency': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'positive_periods': len([r for r in returns if r > 0]) / len(returns),
            'avg_return': np.mean(returns),
            'return_range': max(returns) - min(returns)
        }

    def _analyze_out_of_sample_performance(self, results: Dict) -> Dict:
        """Analyze out-of-sample performance"""
        train_returns = []
        test_returns = []

        for train_result, test_result in zip(results['train_results'], results['test_results']):
            if train_result and 'total_return' in train_result:
                train_returns.append(train_result['total_return'])
            if test_result and 'total_return' in test_result:
                test_returns.append(test_result['total_return'])

        if not train_returns or not test_returns:
            return {}

        return {
            'avg_train_return': np.mean(train_returns),
            'avg_test_return': np.mean(test_returns),
            'train_test_correlation': np.corrcoef(train_returns, test_returns)[0, 1] if len(train_returns) > 1 and len(test_returns) > 1 else 0,
            'overfitting_ratio': np.mean(test_returns) / np.mean(train_returns) if np.mean(train_returns) != 0 else 0
        }

    def _analyze_parameter_importance(self, results: List[Dict], param_names: List[str]) -> Dict:
        """Analyze which parameters have the most impact on performance"""
        if len(results) < 10:
            return {}

        # Simple sensitivity analysis
        importance = {}

        for param in param_names:
            param_values = [r['parameters'][param] for r in results if param in r['parameters']]
            param_returns = [r['total_return'] for r in results if param in r['parameters']]

            if len(set(param_values)) > 1 and len(param_returns) > 1:
                # Calculate correlation between parameter and return
                correlation = np.corrcoef(param_values, param_returns)[0, 1]
                importance[param] = abs(correlation)

        return importance

    def _calculate_robustness_score(self, results: List[Dict]) -> float:
        """Calculate overall robustness score (0-1)"""
        if not results:
            return 0.0

        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]

        # Multi-factor robustness score
        return_score = 1.0 if np.mean(returns) > 0 else 0.0
        sharpe_score = min(1.0, max(0.0, np.mean(sharpes) / 2.0))  # Normalize to 0-1
        drawdown_score = 1.0 if np.mean(drawdowns) < -10 else max(0.0, 1.0 - abs(np.mean(drawdowns)) / 50.0)

        return (return_score * 0.4 + sharpe_score * 0.4 + drawdown_score * 0.2)

    def _apply_stress_scenario(self, symbols: List[str], start_date: str, end_date: str,
                             scenario: Dict) -> Dict:
        """Apply stress scenario modifications to data"""
        # This is a placeholder for stress testing implementation
        # In practice, would modify price data based on scenario parameters
        return {}

    def _run_backtest_with_data(self, data_dict: Dict, strategy_params: Dict = None) -> Dict:
        """Run backtest with pre-loaded data"""
        # Placeholder implementation
        return {}
