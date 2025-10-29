#!/usr/bin/env python3
"""
Simple Trading Algorithm CLI
Run algorithmic trading strategies with public market data
"""
import argparse
import sys
from trading.algorithm import TradingAlgorithm


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Sophisticated Trading Algorithm with Advanced Strategies and Risk Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Conservative approaches
  python main.py --symbols AAPL,MSFT,GOOGL --strategy conservative --risk-profile very_low
  python main.py --symbols TSLA --strategy balanced --risk-profile low

  # Medium risk strategies
  python main.py --symbols SPY,QQQ --strategy enhanced --risk-profile medium
  python main.py --symbols AAPL --strategy ml_style --period 6mo

  # High risk strategies
  python main.py --symbols MSFT,GOOGL --strategy momentum --risk-profile high
  python main.py --symbols TSLA --strategy breakout --risk-profile very_high

  # Very high risk strategies
  python main.py --symbols AAPL --strategy scalping --risk-profile very_high
  python main.py --symbols MSFT --strategy contrarian --risk-profile very_high
  python main.py --symbols GOOGL --strategy leveraged_momentum --risk-profile very_high

  # Advanced strategies (require multiple symbols)
  python main.py --symbols AAPL,MSFT,GOOGL,AMZN --strategy pairs_trading --risk-profile medium
  python main.py --symbols AAPL,MSFT,GOOGL,JNJ --strategy statistical_arbitrage --risk-profile low
  python main.py --symbols AAPL,MSFT,JNJ,XOM --strategy sector_rotation --risk-profile medium
  python main.py --symbols AAPL,MSFT,GOOGL --strategy market_regime --risk-profile medium

  # Research strategies (emerging quantitative approaches)
  python main.py --symbols AAPL,MSFT,GOOGL,AMZN,META --strategy factor_momentum --risk-profile medium
  python main.py --symbols AAPL,MSFT,GOOGL,TSLA,NVDA --strategy cross_sectional_momentum --risk-profile medium
  python main.py --symbols AAPL,MSFT,JNJ,XOM,WMT --strategy liquidity_timing --risk-profile low
        """
    )

    parser.add_argument(
        '--symbols',
        required=True,
        help='Stock symbols to trade (comma-separated, e.g., AAPL,MSFT,GOOGL)'
    )

    parser.add_argument(
        '--strategy',
        choices=['ma', 'rsi', 'momentum', 'mean_reversion', 'breakout',
                'multitimeframe', 'combined', 'enhanced', 'enhanced_combined',
                'scalping', 'contrarian', 'leveraged_momentum', 'ml_style',
                'conservative', 'balanced', 'pairs_trading', 'statistical_arbitrage',
                'sector_rotation', 'market_regime', 'factor_momentum',
                'cross_sectional_momentum', 'research_volatility_regime',
                'liquidity_timing', 'statistical_process_control'],
        default='enhanced',
        help='Trading strategy to use (default: enhanced)'
    )

    parser.add_argument(
        '--risk-profile',
        choices=['very_low', 'low', 'medium', 'high', 'very_high'],
        default='medium',
        help='Risk tolerance level (default: medium)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital for backtest (default: 100000)'
    )

    parser.add_argument(
        '--period',
        default='1y',
        help='Historical data period (default: 1y)'
    )

    parser.add_argument(
        '--output',
        help='Output file for detailed results (CSV format)'
    )

    parser.add_argument(
        '--advanced-metrics',
        action='store_true',
        help='Display advanced performance metrics (Sharpe, Sortino, Calmar ratios)'
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Validate symbols
    if len(symbols) == 0:
        print("Error: No symbols provided")
        sys.exit(1)

    print("=== Sophisticated Trading Algorithm ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Strategy: {args.strategy}")
    print(f"Risk Profile: {args.risk_profile}")
    print(f"Initial Capital: ${args.capital:.2f}")
    print(f"Data Period: {args.period}")
    print()

    # Initialize and run algorithm
    try:
        algorithm = TradingAlgorithm(
            initial_capital=args.capital,
            strategy=args.strategy,
            risk_profile=args.risk_profile
        )

        # Run backtest
        results = algorithm.run_backtest(
            symbols=symbols,
            period=args.period
        )

        if "error" in results:
            print(f"Error: {results['error']}")
            sys.exit(1)

        # Display results
        print("=== Backtest Results ===")
        print(f"Strategy: {results['strategy']}")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")

        if results['total_trades'] > 0:
            print(f"Profit/Loss: ${results['total_profit']:.2f}")

        # Display advanced metrics if available
        if 'sharpe_ratio' in results:
            print()
            print("=== Advanced Performance Metrics ===")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
            print(f"Sortino Ratio: {results.get('sortino_ratio', 0):.4f}")
            print(f"Calmar Ratio: {results.get('calmar_ratio', 0):.4f}")
            print(f"Information Ratio: {results.get('information_ratio', 0):.4f}")
            print(f"Alpha: {results.get('alpha', 0):.4f}")
            print(f"Beta: {results.get('beta', 0):.4f}")
            print(f"Volatility: {results.get('volatility', 0):.4f}")
            print(f"VaR 95%: {results.get('var_95', 0):.4f}")
            print(f"CVaR 95%: {results.get('cvar_95', 0):.4f}")
            print(f"Recovery Factor: {results.get('recovery_factor', 0):.4f}")
            print(f"Profit Factor: {results.get('profit_factor', 0):.4f}")
            print(f"Expectancy: ${results.get('expectancy', 0):.2f}")

            # Performance rating
            sharpe = results.get('sharpe_ratio', 0)
            rating = "5-Star" if sharpe > 1.0 else "4-Star" if sharpe > 0.5 else "3-Star" if sharpe > 0 else "2-Star"
            print(f"Performance Rating: {rating} (Sharpe: {sharpe:.4f})")

        print()
        print("=== Recent Portfolio Values ===")
        recent_history = results['portfolio_history'].tail(10)
        for _, row in recent_history.iterrows():
            print(f"{row['date'].date()}: ${row['total_value']:.2f}")

        # Save detailed results if requested
        if args.output:
            try:
                results['portfolio_history'].to_csv(args.output, index=False)
                print(f"\nDetailed results saved to: {args.output}")
            except Exception as e:
                print(f"Warning: Could not save to {args.output}: {str(e)}")

    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
