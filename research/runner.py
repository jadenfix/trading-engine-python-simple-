"""
Research Framework Runner for Advanced Quantitative Strategies

This module provides tools for running and analyzing the emerging quantitative strategies
and correlation analysis tools in the research framework.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .correlation_analyzer import CorrelationAnalyzer
from .strategies import FactorMomentumStrategy, CrossSectionalMomentumStrategy
from .strategies import VolatilityRegimeStrategy, LiquidityTimingStrategy, StatisticalProcessControlStrategy
from .unconventional_strategies import (
    AttentionDrivenStrategy, SentimentRegimeStrategy, InformationTheoryStrategy,
    ComplexSystemsStrategy, FractalChaosStrategy, QuantumInspiredStrategy
)
from trading.data_fetcher import DataFetcher


class ResearchRunner:
    """Runner for research framework strategies and analysis"""

    def _extract_signals(self, signals, current_date):
        """Extract long and short signals from signal dictionary"""
        long_signals = []
        short_signals = []

        for symbol, signal in signals.items():
            if isinstance(signal, pd.DataFrame):
                # DataFrame signal (from unconventional strategies)
                if current_date in signal.index:
                    signal_value = signal.loc[current_date, 'signal']
                    if signal_value == 1:
                        long_signals.append(symbol)
                    elif signal_value == -1:
                        short_signals.append(symbol)
            else:
                # Scalar signal (from traditional strategies)
                if signal == 1:
                    long_signals.append(symbol)
                elif signal == -1:
                    short_signals.append(symbol)

        return long_signals, short_signals

    def __init__(self):
        """Initialize research runner"""
        self.data_fetcher = DataFetcher()
        self.correlation_analyzer = CorrelationAnalyzer()

    def analyze_correlation_opportunities(self, symbols, period='1y', min_correlation=0.1):
        """
        Analyze correlation opportunities across multiple assets

        Args:
            symbols (list): List of stock symbols to analyze
            period (str): Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            min_correlation (float): Minimum correlation threshold

        Returns:
            dict: Comprehensive correlation analysis
        """
        print(f"Analyzing correlations for {len(symbols)} assets...")

        # Fetch data for all symbols
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data
            else:
                print(f"Warning: No data available for {symbol}")

        if not price_data:
            return {"error": "No data available for any symbols"}

        # Run correlation analysis
        correlation_results = self.correlation_analyzer.analyze_cross_sectional_correlations(price_data)

        # Find statistical arbitrage opportunities
        arb_results = self.correlation_analyzer.find_statistical_arbitrage_opportunities(price_data)

        # Find network correlations
        network_results = self.correlation_analyzer.find_network_correlations(price_data, min_correlation)

        # Find seasonal patterns
        seasonal_results = self.correlation_analyzer.find_seasonal_patterns(price_data)

        return {
            'correlation_analysis': correlation_results,
            'arbitrage_opportunities': arb_results,
            'network_analysis': network_results,
            'seasonal_patterns': seasonal_results,
            'summary': self._generate_correlation_summary(correlation_results, arb_results, network_results)
        }

    def run_factor_momentum_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run factor momentum strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Factor momentum strategy results
        """
        print(f"Running Factor Momentum Strategy on {len(symbols)} assets...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy with shorter lookback for testing
        if strategy_params is None:
            strategy_params = {'lookback_period': 100}  # Shorter lookback for testing

        strategy = FactorMomentumStrategy(**strategy_params)

        # Calculate factor exposures
        factor_exposures = strategy.calculate_factor_exposures(price_data)

        # Calculate factor momentum
        if factor_exposures:
            current_date = max(data.index[-1] for data in price_data.values())
            formation_end = current_date
            formation_start = formation_end - pd.Timedelta(days=strategy.formation_period)

            factor_momentum = strategy.calculate_factor_momentum(factor_exposures, formation_start, formation_end)

            return {
                'factor_exposures': factor_exposures,
                'factor_momentum': factor_momentum,
                'best_factor': max(factor_momentum.items(), key=lambda x: abs(x[1].iloc[-1])) if factor_momentum else None,
                'strategy_params': strategy_params,
                'analysis_date': current_date
            }
        else:
            return {"error": "Insufficient data for factor analysis"}

    def run_cross_sectional_momentum(self, symbols, period='1y', strategy_params=None):
        """
        Run cross-sectional momentum strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Cross-sectional momentum results
        """
        print(f"Running Cross-Sectional Momentum Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = CrossSectionalMomentumStrategy(**strategy_params)

        # Calculate cross-sectional ranks
        current_date = max(data.index[-1] for data in price_data.values())
        momentum_ranks = strategy.calculate_cross_sectional_ranks(price_data, current_date)

        return {
            'momentum_ranks': momentum_ranks,
            'top_performers': [symbol for symbol, rank in momentum_ranks.items() if rank == 1],
            'bottom_performers': [symbol for symbol, rank in momentum_ranks.items() if rank == -1],
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_volatility_regime_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run volatility regime strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Volatility regime analysis results
        """
        print(f"Running Volatility Regime Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = VolatilityRegimeStrategy(**strategy_params)

        # Detect current regime
        current_date = max(data.index[-1] for data in price_data.values())
        current_regime = strategy.detect_volatility_regime(price_data, current_date)

        # Generate signals
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'current_regime': current_regime,
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_liquidity_timing_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run liquidity timing strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Liquidity timing analysis results
        """
        print(f"Running Liquidity Timing Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = LiquidityTimingStrategy(**strategy_params)

        # Calculate liquidity metrics for analysis
        liquidity_analysis = {}
        for symbol, data in price_data.items():
            liquidity_data = strategy.calculate_liquidity_metrics(data)
            current_date = data.index[-1]

            if current_date in liquidity_data.index:
                current_metrics = {
                    'volume_ratio': liquidity_data['volume_ratio'].loc[current_date],
                    'price_impact': liquidity_data['price_impact'].loc[current_date],
                    'amihud': liquidity_data['amihud'].loc[current_date],
                    'turnover': liquidity_data['turnover'].loc[current_date]
                }
                liquidity_analysis[symbol] = current_metrics

        # Generate signals
        current_date = max(data.index[-1] for data in price_data.values())
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'liquidity_analysis': liquidity_analysis,
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_statistical_process_control(self, symbols, period='1y', strategy_params=None):
        """
        Run statistical process control strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Statistical process control results
        """
        print(f"Running Statistical Process Control Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = StatisticalProcessControlStrategy(**strategy_params)

        # Detect regime changes
        current_date = max(data.index[-1] for data in price_data.values())
        regime_indicators = strategy.detect_regime_changes(price_data, current_date)

        # Generate signals
        signals = strategy.generate_signals(price_data, current_date)

        # Calculate control limits for each asset
        control_limits = {}
        for symbol in symbols:
            if symbol in price_data:
                data = price_data[symbol]
                returns = data['Close'].pct_change().dropna()

                if len(returns) >= strategy.min_window:
                    limits = strategy.calculate_control_limits(returns)
                    control_limits[symbol] = limits

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'regime_indicators': regime_indicators,
            'trading_signals': signals,
            'control_limits': control_limits,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_attention_driven_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run attention-driven behavioral strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Attention-driven strategy results
        """
        print(f"Running Attention-Driven Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = AttentionDrivenStrategy(**strategy_params)

        # Detect attention regimes
        current_date = max(data.index[-1] for data in price_data.values())
        attention_regimes = {}

        for symbol in symbols:
            if symbol in price_data:
                data = price_data[symbol]
                if len(data) >= strategy.attention_lookback:
                    attention_data = strategy.calculate_attention_metrics(data)
                    regime = strategy.detect_attention_regimes(attention_data, current_date)
                    attention_regimes[symbol] = regime

        # Generate signals
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'attention_regimes': attention_regimes,
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_sentiment_regime_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run sentiment regime behavioral strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Sentiment regime strategy results
        """
        print(f"Running Sentiment Regime Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = SentimentRegimeStrategy(**strategy_params)

        # Detect sentiment regimes
        current_date = max(data.index[-1] for data in price_data.values())
        sentiment_regimes = {}

        for symbol in symbols:
            if symbol in price_data:
                data = price_data[symbol]
                if len(data) >= strategy.sentiment_lookback:
                    sentiment_data = strategy.calculate_sentiment_indicators(data)
                    regime = strategy.detect_sentiment_regime(sentiment_data, current_date)
                    sentiment_regimes[symbol] = regime

        # Generate signals
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'sentiment_regimes': sentiment_regimes,
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_information_theory_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run information theory strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Information theory strategy results
        """
        print(f"Running Information Theory Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = InformationTheoryStrategy(**strategy_params)

        # Generate signals
        current_date = max(data.index[-1] for data in price_data.values())
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_complex_systems_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run complex systems strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Complex systems strategy results
        """
        print(f"Running Complex Systems Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = ComplexSystemsStrategy(**strategy_params)

        # Generate signals
        current_date = max(data.index[-1] for data in price_data.values())
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_fractal_chaos_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run fractal chaos strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Fractal chaos strategy results
        """
        print(f"Running Fractal Chaos Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = FractalChaosStrategy(**strategy_params)

        # Generate signals
        current_date = max(data.index[-1] for data in price_data.values())
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def run_quantum_inspired_strategy(self, symbols, period='1y', strategy_params=None):
        """
        Run quantum-inspired strategy

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data
            strategy_params (dict): Strategy parameters

        Returns:
            dict: Quantum-inspired strategy results
        """
        print(f"Running Quantum-Inspired Strategy...")

        # Fetch data
        price_data = {}
        for symbol in symbols:
            data = self.data_fetcher.get_historical_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data

        if not price_data:
            return {"error": "No data available"}

        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}

        strategy = QuantumInspiredStrategy(**strategy_params)

        # Generate signals
        current_date = max(data.index[-1] for data in price_data.values())
        signals = strategy.generate_signals(price_data, current_date)

        long_signals, short_signals = self._extract_signals(signals, current_date)

        return {
            'trading_signals': signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'strategy_params': strategy_params,
            'analysis_date': current_date
        }

    def _generate_correlation_summary(self, correlation_results, arb_results, network_results):
        """Generate summary of correlation analysis"""
        summary = {
            'total_assets': len(correlation_results['symbols']),
            'significant_correlations': len(correlation_results['significant_pairs']),
            'cointegrated_pairs': len(correlation_results['cointegration_pairs']),
            'network_clusters': network_results['clusters']['num_clusters'],
            'strongest_correlation': max([pair['correlation'] for pair in correlation_results['significant_pairs']], default=0),
            'best_arbitrage_opportunity': arb_results['best_opportunity']
        }

        return summary

    def run_comprehensive_analysis(self, symbols, period='1y'):
        """
        Run comprehensive analysis using all research tools

        Args:
            symbols (list): List of stock symbols
            period (str): Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            dict: Comprehensive analysis results
        """
        print(f"Running Comprehensive Research Analysis on {len(symbols)} assets...")

        # Run all analysis types
        correlation_analysis = self.analyze_correlation_opportunities(symbols, period=period)
        factor_momentum = self.run_factor_momentum_strategy(symbols, period=period)
        cross_sectional = self.run_cross_sectional_momentum(symbols, period=period)
        volatility_regime = self.run_volatility_regime_strategy(symbols, period=period)
        liquidity_timing = self.run_liquidity_timing_strategy(symbols, period=period)
        spc_analysis = self.run_statistical_process_control(symbols, period=period)

        # Run unconventional strategies
        attention_driven = self.run_attention_driven_strategy(symbols, period=period)
        sentiment_regime = self.run_sentiment_regime_strategy(symbols, period=period)
        information_theory = self.run_information_theory_strategy(symbols, period=period)
        complex_systems = self.run_complex_systems_strategy(symbols, period=period)
        fractal_chaos = self.run_fractal_chaos_strategy(symbols, period=period)
        quantum_inspired = self.run_quantum_inspired_strategy(symbols, period=period)

        return {
            'correlation_analysis': correlation_analysis,
            'factor_momentum': factor_momentum,
            'cross_sectional_momentum': cross_sectional,
            'volatility_regime': volatility_regime,
            'liquidity_timing': liquidity_timing,
            'statistical_process_control': spc_analysis,
            # Unconventional strategies
            'attention_driven': attention_driven,
            'sentiment_regime': sentiment_regime,
            'information_theory': information_theory,
            'complex_systems': complex_systems,
            'fractal_chaos': fractal_chaos,
            'quantum_inspired': quantum_inspired,
            'analysis_summary': self._generate_comprehensive_summary(
                correlation_analysis, factor_momentum, cross_sectional,
                volatility_regime, liquidity_timing, spc_analysis,
                attention_driven, sentiment_regime, information_theory,
                complex_systems, fractal_chaos, quantum_inspired
            )
        }

    def _generate_comprehensive_summary(self, *analysis_results):
        """Generate comprehensive summary across all analyses"""
        summary = {
            'total_strategies_tested': len(analysis_results),
            'trading_opportunities_found': 0,
            'risk_adjusted_opportunities': 0,
            'diversification_benefits': 0
        }

        # Count opportunities from each analysis
        for analysis in analysis_results:
            if 'error' not in analysis:
                # Count significant findings in each analysis
                if 'arbitrage_opportunities' in analysis:
                    arb_opps = analysis['arbitrage_opportunities']
                    if 'significant_pairs' in arb_opps:
                        summary['trading_opportunities_found'] += arb_opps['significant_pairs']

        return summary


def run_research_analysis(symbols=None, analysis_type='comprehensive'):
    """
    Convenience function to run research analysis

    Args:
        symbols (list): List of symbols to analyze (default: major tech stocks)
        analysis_type (str): Type of analysis ('comprehensive', 'correlation', 'factor', 'momentum', 'volatility', 'liquidity', 'spc', 'attention', 'sentiment', 'info_theory', 'complex_systems', 'fractal_chaos', 'quantum')

    Returns:
        dict: Analysis results
    """
    if symbols is None:
        # Default to major tech stocks for demonstration
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
                  'JPM', 'BAC', 'JNJ', 'XOM', 'WMT', 'HD']

    runner = ResearchRunner()

    if analysis_type == 'comprehensive':
        return runner.run_comprehensive_analysis(symbols)
    elif analysis_type == 'correlation':
        return runner.analyze_correlation_opportunities(symbols)
    elif analysis_type == 'factor':
        return runner.run_factor_momentum_strategy(symbols)
    elif analysis_type == 'momentum':
        return runner.run_cross_sectional_momentum(symbols)
    elif analysis_type == 'volatility':
        return runner.run_volatility_regime_strategy(symbols)
    elif analysis_type == 'liquidity':
        return runner.run_liquidity_timing_strategy(symbols)
    elif analysis_type == 'spc':
        return runner.run_statistical_process_control(symbols)
    elif analysis_type == 'attention':
        return runner.run_attention_driven_strategy(symbols)
    elif analysis_type == 'sentiment':
        return runner.run_sentiment_regime_strategy(symbols)
    elif analysis_type == 'info_theory':
        return runner.run_information_theory_strategy(symbols)
    elif analysis_type == 'complex_systems':
        return runner.run_complex_systems_strategy(symbols)
    elif analysis_type == 'fractal_chaos':
        return runner.run_fractal_chaos_strategy(symbols)
    elif analysis_type == 'quantum':
        return runner.run_quantum_inspired_strategy(symbols)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    print("Research Framework Demo")
    print("=" * 50)

    # Run comprehensive analysis
    results = run_research_analysis(symbols, 'comprehensive')

    if 'error' not in results:
        print("\nAnalysis Summary:")
        summary = results['analysis_summary']
        print(f"Total strategies tested: {summary['total_strategies_tested']}")
        print(f"Trading opportunities found: {summary['trading_opportunities_found']}")
        print(f"Diversification benefits identified: {summary['diversification_benefits']}")

        # Show best opportunities from each analysis
        for analysis_name, analysis_result in results.items():
            if analysis_name != 'analysis_summary' and 'error' not in analysis_result:
                print(f"\n{analysis_name.replace('_', ' ').title()}:")
                if analysis_name == 'correlation_analysis':
                    if 'arbitrage_opportunities' in analysis_result:
                        arb = analysis_result['arbitrage_opportunities']
                        if arb['best_opportunity']:
                            best = arb['best_opportunity']
                            print(f"  Best pair: {best['pair']} (correlation: {best['correlation']:.3f})")
                elif analysis_name == 'factor_momentum':
                    if 'best_factor' in analysis_result:
                        factor, momentum = analysis_result['best_factor']
                        print(f"  Strongest factor: {factor} (momentum: {momentum:.4f})")
                elif analysis_name == 'cross_sectional_momentum':
                    if 'top_performers' in analysis_result:
                        print(f"  Top performers: {len(analysis_result['top_performers'])}")
                elif analysis_name == 'volatility_regime':
                    print(f"  Current regime: {analysis_result['current_regime']}")
                elif analysis_name == 'liquidity_timing':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'statistical_process_control':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'attention_driven':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'sentiment_regime':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'information_theory':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'complex_systems':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'fractal_chaos':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
                elif analysis_name == 'quantum_inspired':
                    print(f"  Long signals: {len(analysis_result['long_signals'])}")
    else:
        print(f"Error: {results['error']}")
