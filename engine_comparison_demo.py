#!/usr/bin/env python3
"""
Engine Comparison Demo

This script demonstrates the performance differences between Python, C++, and Rust engines.
Run this to see the dramatic performance improvements with compiled engines.
"""

import time
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(n_symbols: int = 100, n_ticks: int = 1000) -> List[Dict[str, Any]]:
    """Generate test market data"""
    import random

    market_data = []
    base_timestamp = 1234567890000000000  # Nanosecond timestamp

    for symbol_id in range(n_symbols):
        for tick in range(n_ticks):
            # Generate realistic price movements
            base_price = 100.0 + (symbol_id % 50) * 10
            price_noise = random.gauss(0, 0.01)  # 1% volatility
            price = base_price * (1 + price_noise)

            market_data.append({
                'symbol_id': symbol_id,
                'bid_price': ".4f",
                'ask_price': ".4f",
                'bid_size': random.randint(100, 10000),
                'ask_size': random.randint(100, 10000),
                'timestamp': base_timestamp + tick * 1000000,  # 1ms intervals
                'venue_id': random.randint(1, 5),
                'flags': 0,
            })

    logger.info(f"Generated {len(market_data)} market data points for {n_symbols} symbols")
    return market_data

def benchmark_python_engine(market_data: List[Dict[str, Any]], n_runs: int = 5) -> Dict[str, float]:
    """Benchmark Python engine performance"""
    try:
        from research.cpp_integration import CitadelTradingEngine

        logger.info("🔧 Testing Python Engine (fallback)...")

        # Use Python implementation
        engine = CitadelTradingEngine()

        times = []
        total_signals = 0
        total_orders = 0

        for i in range(n_runs):
            start_time = time.perf_counter()

            # Process in batches to simulate real usage
            batch_size = 1000
            for j in range(0, len(market_data), batch_size):
                batch = market_data[j:j + batch_size]
                result = engine.process_market_data(batch, check_risk=False)
                total_signals += len(result.get('signals', []))
                total_orders += len(result.get('orders', []))

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        throughput = len(market_data) * n_runs / avg_time

        return {
            'engine': 'Python',
            'avg_time_seconds': avg_time,
            'throughput_ticks_per_second': throughput,
            'total_signals': total_signals,
            'total_orders': total_orders,
            'available': True,
        }

    except ImportError as e:
        logger.warning(f"Python engine not available: {e}")
        return {
            'engine': 'Python',
            'available': False,
            'error': str(e),
        }

def benchmark_cpp_engine(market_data: List[Dict[str, Any]], n_runs: int = 5) -> Dict[str, float]:
    """Benchmark C++ engine performance"""
    try:
        from research.cpp_integration import CitadelTradingEngine

        logger.info("⚡ Testing C++ Engine...")

        # Configure for maximum performance
        config = {
            'max_orders_per_second': 100000,
            'enable_simd': True,
            'enable_quantization': True,
            'memory_pool_size': 134217728,  # 128MB
        }

        engine = CitadelTradingEngine(config)

        times = []
        total_signals = 0
        total_orders = 0

        for i in range(n_runs):
            start_time = time.perf_counter()

            # Process in larger batches for C++ efficiency
            batch_size = 5000
            for j in range(0, len(market_data), batch_size):
                batch = market_data[j:j + batch_size]
                result = engine.process_market_data(batch, check_risk=False)

                # Extract metrics from C++ result
                total_signals += len(result.get('signals', []))
                total_orders += len(result.get('orders', []))
                processing_time_ms = result.get('processing_latency_ms', 0)

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        throughput = len(market_data) * n_runs / avg_time

        return {
            'engine': 'C++',
            'avg_time_seconds': avg_time,
            'throughput_ticks_per_second': throughput,
            'total_signals': total_signals,
            'total_orders': total_orders,
            'available': True,
        }

    except ImportError as e:
        logger.warning(f"C++ engine not available: {e}")
        return {
            'engine': 'C++',
            'available': False,
            'error': str(e),
        }

def benchmark_rust_engine(market_data: List[Dict[str, Any]], n_runs: int = 5) -> Dict[str, float]:
    """Benchmark Rust engine performance"""
    try:
        from rust_integration import RustTradingEngine

        logger.info("🦀 Testing Rust Engine...")

        engine = RustTradingEngine()

        times = []
        total_signals = 0
        total_orders = 0

        for i in range(n_runs):
            start_time = time.perf_counter()

            # Process all data at once for Rust efficiency
            result = engine.process_market_data(market_data, check_risk=False)

            total_signals += result.signal_count
            total_orders += result.order_count

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        throughput = len(market_data) * n_runs / avg_time

        return {
            'engine': 'Rust',
            'avg_time_seconds': avg_time,
            'throughput_ticks_per_second': throughput,
            'total_signals': total_signals,
            'total_orders': total_orders,
            'available': True,
            'processing_time_us': result.processing_time_us,
        }

    except ImportError as e:
        logger.warning(f"⚠️  Rust engine not available: {e}")
        logger.warning("To enable Rust engine, run: ./build_rust.sh")
        return {
            'engine': 'Rust',
            'available': False,
            'error': str(e),
        }

def print_results(results: Dict[str, Dict[str, Any]]):
    """Print benchmark results in a nice format"""

    print("\n" + "="*80)
    print("🚀 ENGINE PERFORMANCE COMPARISON")
    print("="*80)

    # Header
    print("<12")
    print("-" * 70)

    # Results
    for engine_name in ['Python', 'C++', 'Rust']:
        if engine_name in results:
            result = results[engine_name]
            if result.get('available', False):
                print("<12"
                      "<8.2f"
                      "<12.0f"
                      "<8d"
                      "<8d")
            else:
                print("<12"
                      "<8"
                      "<12"
                      "<8"
                      "<8")

    print("-" * 70)

    # Performance ratios
    if all(results.get(engine, {}).get('available', False) for engine in ['Python', 'C++', 'Rust']):
        python_time = results['Python']['avg_time_seconds']
        cpp_time = results['C++']['avg_time_seconds']
        rust_time = results['Rust']['avg_time_seconds']

        print("\nPERFORMANCE RATIOS:")
        print(".1f")
        print(".1f")
        print(".1f")

        print("\nTHROUGHPUT RATIOS:")
        python_throughput = results['Python']['throughput_ticks_per_second']
        cpp_throughput = results['C++']['throughput_ticks_per_second']
        rust_throughput = results['Rust']['throughput_ticks_per_second']

        print(".0f")
        print(".0f")
        print(".0f")

def main():
    """Main benchmark function"""
    print("🎯 Citadel Trading Engine - Performance Benchmark")
    print("Compares Python, C++, and Rust engine performance")
    print()

    # Generate test data
    market_data = generate_test_data(n_symbols=50, n_ticks=500)  # 25,000 total ticks

    # Benchmark engines
    results = {}

    # Python benchmark (always available as fallback)
    results['Python'] = benchmark_python_engine(market_data, n_runs=3)

    # C++ benchmark
    results['C++'] = benchmark_cpp_engine(market_data, n_runs=3)

    # Rust benchmark (recommended)
    results['Rust'] = benchmark_rust_engine(market_data, n_runs=3)

    # Print results
    print_results(results)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if results.get('Rust', {}).get('available', False):
        print("  ✅ Rust engine is available and recommended for production use")
        print("     - Best performance and memory safety")
        print("     - Fearless concurrency and modern tooling")
    elif results.get('C++', {}).get('available', False):
        print("  ⚡ C++ engine is available - good performance but requires careful memory management")
    else:
        print("  🐍 Only Python engine available - good for development and testing")
        print("     - Consider building Rust or C++ engines for production performance")

    print("\n🔧 To build engines:")
    print("  Rust: ./build_rust.sh")
    print("  C++:  ./build_cpp.sh")

if __name__ == "__main__":
    main()
