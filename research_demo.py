#!/usr/bin/env python3
"""
Advanced Quantitative Research Framework Demo

This script demonstrates the full capabilities of our enhanced research framework
including unconventional strategies, ensemble allocation, and comprehensive backtesting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add research directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'research'))

from research.runner import run_research_analysis
from research.backtesting_engine import BacktestingEngine, run_strategy_backtest
from research.strategy_ensemble import StrategyEnsemble, create_unconventional_ensemble


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def demo_unconventional_strategies():
    """Demonstrate unconventional strategy capabilities"""
    print_header("UNCONVENTIONAL QUANTITATIVE STRATEGIES DEMO")

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

    print(f"Testing strategies on symbols: {symbols}")
    print("\nAvailable unconventional strategies:")
    print("• Attention-Driven Strategy (behavioral finance)")
    print("• Sentiment Regime Strategy (market psychology)")
    print("• Information Theory Strategy (complexity measures)")
    print("• Complex Systems Strategy (network effects)")
    print("• Fractal Chaos Strategy (fractal geometry)")
    print("• Quantum-Inspired Strategy (quantum mechanics concepts)")

    # Test each strategy
    strategies = ['attention', 'sentiment', 'info_theory', 'complex_systems', 'fractal_chaos', 'quantum']

    for strategy in strategies:
        print(f"\nTesting {strategy.replace('_', ' ').title()} Strategy...")
        try:
            result = run_research_analysis(symbols[:2], strategy)  # Use fewer symbols for speed
            if 'error' not in result:
                long_signals = len(result.get('long_signals', []))
                short_signals = len(result.get('short_signals', []))
                print(f"  Success! Long signals: {long_signals}, Short signals: {short_signals}")
            else:
                print(f"  Error: {result['error']}")
        except Exception as e:
            print(f"  Exception: {e}")


def demo_strategy_ensemble():
    """Demonstrate the strategy ensemble system"""
    print_header("STRATEGY ENSEMBLE SYSTEM DEMO")

    try:
        # Create ensemble
        ensemble = create_unconventional_ensemble()

        summary = ensemble.get_ensemble_summary()

        print(f"Ensemble Configuration:")
        print(f"• Total Strategies: {summary['total_strategies']}")
        print(f"• Rebalancing: Daily regime-based allocation")
        print(f"• Risk Parity: Enabled")
        print(f"• Max Strategy Weight: 25%")

        print(f"\nStrategy Categories:")
        for name, category in summary['strategy_categories'].items():
            print(f"• {name}: {category}")

        print(f"\nCurrent Weights:")
        for strategy, weight in summary['current_weights'].items():
            print(".1%")

        print("\nEnsemble system ready for dynamic allocation!")

    except Exception as e:
        print(f"Ensemble demo failed: {e}")


def demo_backtesting_framework():
    """Demonstrate the backtesting framework"""
    print_header("ADVANCED BACKTESTING FRAMEWORK DEMO")

    try:
        # Initialize backtesting engine
        backtest_engine = BacktestingEngine(
            initial_capital=100000,
            commission_per_trade=0.001,
            slippage_bps=5,
            max_position_size=0.1
        )

        print(f"Backtesting Engine Configuration:")
        print(f"• Initial Capital: ${backtest_engine.initial_capital:,.0f}")
        print(f"• Commission: {backtest_engine.commission_per_trade:.2%}")
        print(f"• Slippage: {backtest_engine.slippage_bps} bps")
        print(f"• Max Position Size: {backtest_engine.max_position_size:.1%}")
        print(f"• Risk-Free Rate: {backtest_engine.risk_free_rate:.1%}")

        print("\nBacktesting engine initialized successfully!")
        print("Ready for realistic strategy evaluation with transaction costs and slippage.")

    except Exception as e:
        print(f"Backtesting demo failed: {e}")


def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis capabilities"""
    print_header("COMPREHENSIVE RESEARCH ANALYSIS DEMO")

    symbols = ['AAPL', 'MSFT', 'GOOGL']

    print(f"Running comprehensive analysis on: {symbols}")

    try:
        results = run_research_analysis(symbols, 'comprehensive')

        if 'error' not in results:
            print("\nAnalysis Summary:")

            # Count total signals across all strategies
            total_long = 0
            total_short = 0
            strategy_count = 0

            for analysis_name, analysis_result in results.items():
                if analysis_name != 'analysis_summary' and 'error' not in analysis_result:
                    strategy_count += 1
                    if 'long_signals' in analysis_result:
                        total_long += len(analysis_result['long_signals'])
                    if 'short_signals' in analysis_result:
                        total_short += len(analysis_result['short_signals'])

            print(f"• Strategies Analyzed: {strategy_count}")
            print(f"• Total Long Signals: {total_long}")
            print(f"• Total Short Signals: {total_short}")

            print("\nKey Findings:")
            for analysis_name, analysis_result in results.items():
                if analysis_name != 'analysis_summary' and 'error' not in analysis_result:
                    status = "Active" if ('long_signals' in analysis_result and len(analysis_result['long_signals']) > 0) or ('short_signals' in analysis_result and len(analysis_result['short_signals']) > 0) else "Neutral"
                    print(f"• {analysis_name.replace('_', ' ').title()}: {status}")

        else:
            print(f"Analysis failed: {results['error']}")

    except Exception as e:
        print(f"Comprehensive analysis failed: {e}")


def demo_research_capabilities():
    """Showcase all research capabilities"""
    print_header("COMPLETE RESEARCH FRAMEWORK CAPABILITIES")

    capabilities = [
        "Behavioral Finance Strategies",
        "  - Investor attention modeling",
        "  - Market sentiment regimes",
        "  - Herding behavior detection",
        "  - Anchoring bias exploitation",

        "Information Theory Approaches",
        "  - Approximate entropy measures",
        "  - Transfer entropy analysis",
        "  - Mutual information signals",
        "  - Complexity-based trading",

        "Complex Systems Methods",
        "  - Network centrality analysis",
        "  - Contagion effect detection",
        "  - Synchronization patterns",
        "  - Systemic risk indicators",

        "Chaos Theory Applications",
        "  - Fractal dimension analysis",
        "  - Hurst exponent calculations",
        "  - Lyapunov exponent measures",
        "  - Non-linear dynamics",

        "Quantum-Inspired Concepts",
        "  - Market state superposition",
        "  - Quantum coherence measures",
        "  - Wave function probabilities",
        "  - Entanglement correlations",

        "Advanced Risk Management",
        "  - CVaR optimization",
        "  - Drawdown control",
        "  - Kelly criterion sizing",
        "  - Portfolio stress testing",

        "Ensemble Allocation Systems",
        "  - Regime-dependent weighting",
        "  - Risk parity allocation",
        "  - Performance momentum",
        "  - Dynamic rebalancing",

        "Comprehensive Backtesting",
        "  - Realistic transaction costs",
        "  - Multiple slippage models",
        "  - Position size optimization",
        "  - Performance attribution",

        "Correlation Analysis Tools",
        "  - Cross-sectional correlations",
        "  - Statistical arbitrage pairs",
        "  - Network correlation clusters",
        "  - Seasonal pattern detection",

        "Factor Momentum Models",
        "  - Multi-factor exposure analysis",
        "  - Factor momentum timing",
        "  - Cross-sectional factor ranks",
        "  - Factor rotation strategies"
    ]

    for capability in capabilities:
        print(capability)


def main():
    """Main demo function"""
    print("ADVANCED QUANTITATIVE RESEARCH FRAMEWORK")
    print("Enhanced with Unconventional Strategies & Alpha Generation")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all demos
    demo_unconventional_strategies()
    demo_strategy_ensemble()
    demo_backtesting_framework()
    demo_comprehensive_analysis()
    demo_research_capabilities()

    print_header("DEMO COMPLETED")
    print("Your research framework now includes:")
    print("• 6+ unconventional quantitative strategies")
    print("• Advanced ensemble allocation system")
    print("• Comprehensive backtesting with realistic costs")
    print("• Multi-dimensional correlation analysis")
    print("• Behavioral finance and complexity theory approaches")
    print("\nReady to generate alpha through innovative quantitative methods! 🚀")


if __name__ == "__main__":
    main()
