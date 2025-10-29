"""
Adaptive Parameter Optimization for Quantitative Strategies

This module implements advanced parameter optimization techniques including:
- Bayesian optimization
- Walk-forward analysis
- Genetic algorithms
- Regime-adaptive parameters
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class BayesianOptimizer:
    """Bayesian optimization for strategy parameters"""

    def __init__(self, bounds, n_initial=5, n_iterations=15, random_state=42):
        """
        Initialize Bayesian optimizer

        Args:
            bounds (dict): Parameter bounds {'param_name': (min, max)}
            n_initial (int): Number of initial random evaluations
            n_iterations (int): Number of optimization iterations
            random_state (int): Random state for reproducibility
        """
        self.bounds = bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Track evaluations
        self.X_observed = []
        self.y_observed = []

        # GP model
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            random_state=random_state
        )

    def _sample_initial(self):
        """Sample initial parameter combinations"""
        np.random.seed(self.random_state)
        initial_points = []

        for _ in range(self.n_initial):
            point = {}
            for param, (min_val, max_val) in self.bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    point[param] = np.random.randint(min_val, max_val + 1)
                else:
                    point[param] = np.random.uniform(min_val, max_val)
            initial_points.append(point)

        return initial_points

    def _expected_improvement(self, X):
        """
        Calculate expected improvement acquisition function

        Args:
            X (array): Parameter values to evaluate

        Returns:
            float: Expected improvement value
        """
        if len(self.y_observed) == 0:
            return 1.0

        X = np.array(X).reshape(1, -1)
        mu, sigma = self.gp.predict(X, return_std=True)

        mu = mu[0]
        sigma = sigma[0]

        if sigma == 0:
            return 0

        # Best observed value
        y_best = np.max(self.y_observed)

        # Expected improvement
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei if not np.isnan(ei) else 0

    def _optimize_acquisition(self):
        """Optimize the acquisition function to find next point to evaluate"""
        n_params = len(self.bounds)
        bounds_list = [(min_val, max_val) for min_val, max_val in self.bounds.values()]

        # Start from best known point
        if self.X_observed:
            x0 = self.X_observed[np.argmax(self.y_observed)]
        else:
            x0 = np.array([(min_val + max_val) / 2 for min_val, max_val in self.bounds.values()])

        # Minimize negative expected improvement
        result = minimize(
            lambda x: -self._expected_improvement(x),
            x0=x0,
            bounds=bounds_list,
            method='L-BFGS-B'
        )

        if result.success:
            return result.x
        else:
            # Fallback to random point
            return np.array([np.random.uniform(min_val, max_val) for min_val, max_val in self.bounds.values()])

    def optimize(self, objective_function, verbose=True):
        """
        Run Bayesian optimization

        Args:
            objective_function (callable): Function to maximize, takes parameter dict and returns score
            verbose (bool): Whether to print progress

        Returns:
            dict: Best parameters and optimization history
        """
        # Sample initial points
        initial_points = self._sample_initial()

        if verbose:
            print(f"Running Bayesian optimization with {len(initial_points)} initial points and {self.n_iterations} iterations...")

        # Evaluate initial points
        for i, params in enumerate(initial_points):
            score = objective_function(params)
            self.X_observed.append(list(params.values()))
            self.y_observed.append(score)

            if verbose:
                param_str = ", ".join([f"{k}={v:.3f}" for k, v in params.items()])
                print(f"Initial point {i+1}: {param_str} -> Score: {score:.4f}")

        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Fit GP model
            if len(self.X_observed) >= 2:
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

            # Find next point to evaluate
            next_x = self._optimize_acquisition()
            next_params = dict(zip(self.bounds.keys(), next_x))

            # Evaluate
            score = objective_function(next_params)
            self.X_observed.append(next_x)
            self.y_observed.append(score)

            if verbose:
                param_str = ", ".join([f"{k}={v:.3f}" for k, v in next_params.items()])
                print(f"Iteration {iteration+1}: {param_str} -> Score: {score:.4f}")

        # Find best parameters
        best_idx = np.argmax(self.y_observed)
        best_params = dict(zip(self.bounds.keys(), self.X_observed[best_idx]))
        best_score = self.y_observed[best_idx]

        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': {
                'parameters': self.X_observed,
                'scores': self.y_observed
            }
        }


class WalkForwardOptimizer:
    """Walk-forward optimization with out-of-sample testing"""

    def __init__(self, optimization_window=252, validation_window=63, step_size=21):
        """
        Initialize walk-forward optimizer

        Args:
            optimization_window (int): Days for in-sample optimization
            validation_window (int): Days for out-of-sample validation
            step_size (int): Days to advance for each iteration
        """
        self.optimization_window = optimization_window
        self.validation_window = validation_window
        self.step_size = step_size

    def optimize_strategy(self, strategy_class, parameter_bounds, price_data_dict,
                         start_date, end_date, metric='sharpe_ratio'):
        """
        Run walk-forward optimization

        Args:
            strategy_class: Strategy class to optimize
            parameter_bounds (dict): Parameter bounds for optimization
            price_data_dict (dict): Price data dictionary
            start_date (pd.Timestamp): Start date
            end_date (pd.Timestamp): End date
            metric (str): Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')

        Returns:
            dict: Optimization results
        """
        from .backtesting_engine import BacktestingEngine

        backtest_engine = BacktestingEngine(initial_capital=100000, commission_per_trade=0.001)

        current_date = start_date
        optimization_results = []
        validation_results = []

        while current_date + pd.Timedelta(days=self.optimization_window + self.validation_window) <= end_date:

            # In-sample optimization period
            opt_start = current_date
            opt_end = current_date + pd.Timedelta(days=self.optimization_window)

            # Out-of-sample validation period
            val_start = opt_end
            val_end = min(val_start + pd.Timedelta(days=self.validation_window), end_date)

            if verbose:
                print(f"Walk-forward iteration: {opt_start.date()} to {val_end.date()}")

            # Optimize parameters on in-sample data
            def objective(params):
                try:
                    # Generate signals
                    signals = {}
                    test_dates = pd.date_range(opt_start, opt_end, freq='D')

                    for test_date in test_dates:
                        try:
                            daily_signals = strategy_class(**params).generate_signals(price_data_dict, test_date)
                            for symbol, signal_df in daily_signals.items():
                                if symbol not in signals:
                                    signals[symbol] = signal_df
                        except:
                            continue

                    if not signals:
                        return -999

                    # Run backtest
                    results = backtest_engine.run_backtest(signals, price_data_dict, opt_start, opt_end)

                    # Return metric to maximize
                    if metric == 'sharpe_ratio':
                        return results.get('sharpe_ratio', -999)
                    elif metric == 'total_return':
                        return results.get('total_return', -999)
                    elif metric == 'max_drawdown':
                        return -results.get('max_drawdown', 999)  # Minimize drawdown = maximize negative drawdown
                    else:
                        return results.get('sharpe_ratio', -999)

                except:
                    return -999

            # Run optimization
            optimizer = BayesianOptimizer(parameter_bounds, n_initial=3, n_iterations=8)
            opt_result = optimizer.optimize(objective, verbose=False)

            best_params = opt_result['best_parameters']

            # Validate on out-of-sample data
            signals = {}
            val_dates = pd.date_range(val_start, val_end, freq='D')

            for test_date in val_dates:
                try:
                    daily_signals = strategy_class(**best_params).generate_signals(price_data_dict, test_date)
                    for symbol, signal_df in daily_signals.items():
                        if symbol not in signals:
                            signals[symbol] = signal_df
                except:
                    continue

            if signals:
                val_results = backtest_engine.run_backtest(signals, price_data_dict, val_start, val_end)

                optimization_results.append({
                    'period': f"{opt_start.date()} to {opt_end.date()}",
                    'best_params': best_params,
                    'in_sample_score': opt_result['best_score'],
                    'out_sample_score': val_results.get('sharpe_ratio', 0)
                })

                validation_results.append(val_results)

            # Move to next period
            current_date += pd.Timedelta(days=self.step_size)

        # Calculate overall performance
        if validation_results:
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in validation_results])
            avg_return = np.mean([r.get('total_return', 0) for r in validation_results])
            avg_drawdown = np.mean([r.get('max_drawdown', 0) for r in validation_results])
        else:
            avg_sharpe = avg_return = avg_drawdown = 0

        return {
            'walk_forward_results': optimization_results,
            'overall_performance': {
                'average_sharpe': avg_sharpe,
                'average_return': avg_return,
                'average_max_drawdown': avg_drawdown,
                'total_iterations': len(optimization_results)
            },
            'final_parameters': best_params if optimization_results else None
        }


class AdaptiveParameterManager:
    """Manages adaptive parameters that change based on market regime"""

    def __init__(self):
        """Initialize adaptive parameter manager"""
        self.regime_parameters = {}
        self.parameter_history = []

    def add_regime_parameters(self, regime_name, strategy_name, parameters):
        """
        Add regime-specific parameters

        Args:
            regime_name (str): Market regime name
            strategy_name (str): Strategy name
            parameters (dict): Parameters for this regime
        """
        if regime_name not in self.regime_parameters:
            self.regime_parameters[regime_name] = {}

        self.regime_parameters[regime_name][strategy_name] = parameters

    def get_adaptive_parameters(self, current_regime, strategy_name, default_params=None):
        """
        Get parameters adapted to current regime

        Args:
            current_regime (str): Current market regime
            strategy_name (str): Strategy name
            default_params (dict): Default parameters if regime-specific not available

        Returns:
            dict: Adapted parameters
        """
        if current_regime in self.regime_parameters and strategy_name in self.regime_parameters[current_regime]:
            adapted_params = self.regime_parameters[current_regime][strategy_name].copy()
        else:
            adapted_params = default_params.copy() if default_params else {}

        # Record parameter usage
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'regime': current_regime,
            'strategy': strategy_name,
            'parameters': adapted_params
        })

        return adapted_params

    def update_regime_parameters(self, regime_name, strategy_name, performance_data):
        """
        Update parameters based on performance feedback

        Args:
            regime_name (str): Market regime
            strategy_name (str): Strategy name
            performance_data (dict): Recent performance metrics
        """
        # Simple parameter adaptation based on performance
        if regime_name not in self.regime_parameters:
            return

        if strategy_name not in self.regime_parameters[regime_name]:
            return

        current_params = self.regime_parameters[regime_name][strategy_name]
        sharpe = performance_data.get('sharpe_ratio', 0)

        # Adapt parameters based on Sharpe ratio
        if sharpe > 1.0:
            # Good performance - keep similar parameters
            pass
        elif sharpe > 0.5:
            # Moderate performance - slight adjustments
            for param in current_params:
                if isinstance(current_params[param], (int, float)):
                    # Add small random adjustment
                    adjustment = np.random.normal(0, 0.1)
                    current_params[param] *= (1 + adjustment)
                    # Keep within reasonable bounds
                    current_params[param] = max(0.1, min(10.0, current_params[param]))
        else:
            # Poor performance - more significant changes
            for param in current_params:
                if isinstance(current_params[param], (int, float)):
                    # Larger random adjustment
                    adjustment = np.random.normal(0, 0.3)
                    current_params[param] *= (1 + adjustment)
                    # Keep within bounds
                    current_params[param] = max(0.01, min(100.0, current_params[param]))


def optimize_strategy_parameters(strategy_class, parameter_bounds, price_data_dict,
                               start_date, end_date, optimization_method='bayesian'):
    """
    Convenience function for strategy parameter optimization

    Args:
        strategy_class: Strategy class to optimize
        parameter_bounds (dict): Parameter bounds
        price_data_dict (dict): Price data
        start_date (pd.Timestamp): Start date
        end_date (pd.Timestamp): End date
        optimization_method (str): 'bayesian' or 'walk_forward'

    Returns:
        dict: Optimization results
    """
    if optimization_method == 'bayesian':
        def objective(params):
            try:
                backtest_engine = BacktestingEngine(initial_capital=100000)

                # Generate signals for optimization period
                signals = {}
                test_dates = pd.date_range(start_date, end_date, freq='D')

                for test_date in test_dates:
                    try:
                        daily_signals = strategy_class(**params).generate_signals(price_data_dict, test_date)
                        for symbol, signal_df in daily_signals.items():
                            if symbol not in signals:
                                signals[symbol] = signal_df
                    except:
                        continue

                if not signals:
                    return -999

                results = backtest_engine.run_backtest(signals, price_data_dict, start_date, end_date)
                return results.get('sharpe_ratio', -999)

            except:
                return -999

        optimizer = BayesianOptimizer(parameter_bounds)
        return optimizer.optimize(objective)

    elif optimization_method == 'walk_forward':
        optimizer = WalkForwardOptimizer()
        return optimizer.optimize_strategy(strategy_class, parameter_bounds,
                                         price_data_dict, start_date, end_date)

    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")


# Example usage and testing
if __name__ == "__main__":
    print("Adaptive Parameter Optimization System")
    print("=" * 50)

    # Test Bayesian optimizer with a simple function
    def test_function(params):
        x, y = params['x'], params['y']
        # Rosenbrock function (has global minimum at (1,1))
        return -(100 * (y - x**2)**2 + (1 - x)**2)  # Negative for maximization

    bounds = {'x': (-2, 2), 'y': (-1, 3)}

    optimizer = BayesianOptimizer(bounds, n_initial=3, n_iterations=5)
    result = optimizer.optimize(test_function)

    print(f"Best parameters: {result['best_parameters']}")
    print(f"Best score: {result['best_score']:.4f}")

    print("\nAdaptive optimization system ready!")
