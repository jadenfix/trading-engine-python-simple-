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
        description='Simple Trading Algorithm using public market data APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbols AAPL,MSFT,GOOGL --strategy ma --capital 10000
  python main.py --symbols TSLA --strategy rsi --period 6mo
  python main.py --symbols SPY,QQQ --strategy momentum --capital 50000
  python main.py --symbols AAPL --strategy mean_reversion --period 1y
  python main.py --symbols MSFT,GOOGL --strategy breakout --capital 25000
  python main.py --symbols TSLA --strategy multitimeframe --period 6mo
  python main.py --symbols AAPL,MSFT,GOOGL --strategy enhanced --capital 100000
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
                'multitimeframe', 'combined', 'enhanced', 'enhanced_combined'],
        default='enhanced',
        help='Trading strategy to use (default: enhanced)'
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

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Validate symbols
    if len(symbols) == 0:
        print("Error: No symbols provided")
        sys.exit(1)

    print("=== Simple Trading Algorithm ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Strategy: {args.strategy}")
    print(f"Initial Capital: ${args.capital:.2f}")
    print(f"Data Period: {args.period}")
    print()

    # Initialize and run algorithm
    try:
        algorithm = TradingAlgorithm(
            initial_capital=args.capital,
            strategy=args.strategy
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
