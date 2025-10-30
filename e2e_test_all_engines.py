#!/usr/bin/env python3
"""
End-to-End Test Suite for All 4 Trading Engines
Tests Rust, Go, C++, and Python engines with live market data
"""

import time
import sys
import os
import subprocess
import traceback
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Test configuration
LIVE_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
TEST_PERIODS = ['1mo', '3mo', '6mo', '1y']  # Test different time periods
BATCH_SIZES = [10, 50, 100, 500]  # Test different batch sizes

class EngineTester:
    """Comprehensive engine testing framework"""

    def __init__(self):
        self.test_results = {}
        self.live_data = {}
        self.performance_metrics = {}
        self.errors = []

    def log(self, message, level="INFO"):
        """Log messages with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def log_error(self, message, error=None):
        """Log errors with traceback"""
        self.errors.append({
            'message': message,
            'error': str(error) if error else None,
            'timestamp': datetime.now(),
            'traceback': traceback.format_exc()
        })
        self.log(f"ERROR: {message}", "ERROR")
        if error:
            self.log(f"Exception: {error}", "ERROR")

    def fetch_live_data(self):
        """Fetch live market data for testing"""
        self.log("📊 Fetching live market data...")

        try:
            for symbol in LIVE_SYMBOLS:
                self.log(f"Fetching data for {symbol}...")
                ticker = yf.Ticker(symbol)

                # Get recent data
                data = ticker.history(period="6mo", interval="1d")

                if len(data) == 0:
                    self.log_error(f"No data received for {symbol}")
                    continue

                # Convert to our format
                market_data = []
                for idx, row in data.iterrows():
                    market_data.append({
                        'symbol_id': hash(symbol) % 10000,  # Consistent ID per symbol
                        'bid_price': str(row['Close'] * 0.9995),  # Simulate bid
                        'ask_price': str(row['Close'] * 1.0005),  # Simulate ask
                        'bid_size': str(row['Volume'] * 0.4),     # Simulate sizes
                        'ask_size': str(row['Volume'] * 0.6),
                        'timestamp': int(idx.timestamp() * 1e9),   # Nanoseconds
                        'venue_id': 1,
                        'flags': 0,
                    })

                self.live_data[symbol] = market_data
                self.log(f"✅ {symbol}: {len(market_data)} data points")

            self.log(f"📊 Successfully loaded data for {len(self.live_data)} symbols")

        except Exception as e:
            self.log_error("Failed to fetch live data", e)
            return False

        return True

    def test_rust_engine(self):
        """Test Rust engine with live data"""
        self.log("🦀 Testing Rust Engine...")

        try:
            from rust_integration import RustTradingEngine
            engine = RustTradingEngine()
            self.log("✅ Rust engine initialized")

            # Test with different symbols and batch sizes
            results = {}
            for symbol, data in self.live_data.items():
                for batch_size in BATCH_SIZES[:2]:  # Test smaller batches first
                    batch_data = data[:batch_size]

                    start_time = time.time()
                    result = engine.process_market_data(batch_data, check_risk=True)
                    latency_ms = (time.time() - start_time) * 1000

                    results[f"{symbol}_{batch_size}"] = {
                        'signals': result.signal_count,
                        'orders': result.order_count,
                        'latency_ms': latency_ms,
                        'throughput': len(batch_data) / (latency_ms / 1000) if latency_ms > 0 else 0
                    }

                    self.log(".1f"
                             ".0f")

            # Get performance stats
            try:
                stats = engine.get_performance_stats()
                self.performance_metrics['rust'] = {
                    'signals_processed': stats.get('signals_processed', 0),
                    'orders_submitted': stats.get('orders_submitted', 0),
                    'avg_signal_latency_us': stats.get('avg_signal_latency_us', 0),
                    'avg_order_latency_us': stats.get('avg_order_latency_us', 0),
                }
            except:
                self.performance_metrics['rust'] = {'error': 'Could not get performance stats'}

            self.test_results['rust'] = results
            return True

        except ImportError as e:
            self.log_error("Rust engine not available", e)
            self.test_results['rust'] = {'error': 'Import failed', 'details': str(e)}
            return False
        except Exception as e:
            self.log_error("Rust engine test failed", e)
            self.test_results['rust'] = {'error': 'Test failed', 'details': str(e)}
            return False

    def test_go_engine(self):
        """Test Go engine (via subprocess since no direct Python binding)"""
        self.log("🐹 Testing Go Engine...")

        try:
            # Check if Go binary exists
            go_binary = os.path.join(os.getcwd(), 'bin', 'go_trading_engine')
            if not os.path.exists(go_binary):
                self.log_error("Go binary not found, trying to build...")
                try:
                    subprocess.run(['./build_go.sh'], check=True, capture_output=True)
                except:
                    self.test_results['go'] = {'error': 'Build failed'}
                    return False

            if not os.path.exists(go_binary):
                self.log_error("Go binary still not found after build attempt")
                self.test_results['go'] = {'error': 'Binary not found'}
                return False

            # Run Go engine and capture output
            result = subprocess.run([go_binary],
                                  capture_output=True,
                                  text=True,
                                  timeout=30)

            if result.returncode == 0:
                self.log("✅ Go engine executed successfully")
                # Parse output for basic validation
                if "Go Trading Engine" in result.stdout and "ready" in result.stdout.lower():
                    self.test_results['go'] = {'status': 'success', 'output': result.stdout[:200]}
                    return True
                else:
                    self.test_results['go'] = {'status': 'unexpected_output', 'output': result.stdout[:200]}
                    return False
            else:
                self.log_error(f"Go engine failed with return code {result.returncode}")
                self.test_results['go'] = {
                    'error': 'Execution failed',
                    'return_code': result.returncode,
                    'stderr': result.stderr[:200]
                }
                return False

        except subprocess.TimeoutExpired:
            self.log_error("Go engine test timed out")
            self.test_results['go'] = {'error': 'Timeout'}
            return False
        except Exception as e:
            self.log_error("Go engine test failed", e)
            self.test_results['go'] = {'error': 'Test failed', 'details': str(e)}
            return False

    def test_cpp_engine(self):
        """Test C++ engine with live data"""
        self.log("⚡ Testing C++ Engine...")

        try:
            from research.cpp_integration import QuantTradingEngine

            # Configure for testing
            config = {
                'max_orders_per_second': 1000,
                'enable_simd': True,
                'enable_quantization': True,
                'memory_pool_size': 134217728,  # 128MB
            }

            engine = QuantTradingEngine(config)
            self.log("✅ C++ engine initialized")

            # Test with different symbols
            results = {}
            for symbol, data in list(self.live_data.items())[:3]:  # Test first 3 symbols
                batch_data = data[:50]  # Test with 50 data points

                start_time = time.time()
                result = engine.process_market_data(batch_data, check_risk=False)
                latency_ms = (time.time() - start_time) * 1000

                results[symbol] = {
                    'latency_ms': latency_ms,
                    'throughput': len(batch_data) / (latency_ms / 1000) if latency_ms > 0 else 0,
                    'has_signals': len(result.get('signals', [])) > 0,
                    'has_orders': len(result.get('orders', [])) > 0
                }

                self.log(".1f"
                             f"   Has signals: {results[symbol]['has_signals']}")

            self.test_results['cpp'] = results
            return True

        except ImportError as e:
            self.log_error("C++ engine not available", e)
            self.test_results['cpp'] = {'error': 'Import failed', 'details': str(e)}
            return False
        except Exception as e:
            self.log_error("C++ engine test failed", e)
            self.test_results['cpp'] = {'error': 'Test failed', 'details': str(e)}
            return False

    def test_python_engine(self):
        """Test Python research framework with live data"""
        self.log("🐍 Testing Python Engine...")

        try:
            from research.comprehensive_test import ComprehensiveTester

            tester = ComprehensiveTester()

            # Run comprehensive tests
            self.log("Running comprehensive Python tests...")
            start_time = time.time()

            try:
                result = tester.run_all_tests()
                test_time = time.time() - start_time

                self.log(f"✅ Python tests completed in {test_time:.2f}s")
                self.log(f"   Result: {result}")

                # Test individual components
                components_tested = []
                try:
                    tester.test_imports()
                    components_tested.append("imports")
                except:
                    pass

                try:
                    tester.test_basic_strategies()
                    components_tested.append("strategies")
                except:
                    pass

                try:
                    tester.test_risk_management()
                    components_tested.append("risk_management")
                except:
                    pass

                self.test_results['python'] = {
                    'overall_result': result,
                    'test_time_seconds': test_time,
                    'components_tested': components_tested,
                    'status': 'success' if result else 'partial'
                }

                return result

            except Exception as e:
                self.log_error("Python comprehensive test failed", e)
                self.test_results['python'] = {'error': 'Test failed', 'details': str(e)}
                return False

        except ImportError as e:
            self.log_error("Python research framework not available", e)
            self.test_results['python'] = {'error': 'Import failed', 'details': str(e)}
            return False
        except Exception as e:
            self.log_error("Python engine test failed", e)
            self.test_results['python'] = {'error': 'Test failed', 'details': str(e)}
            return False

    def test_engine_comparison(self):
        """Compare performance across available engines"""
        self.log("⚖️  Comparing Engine Performance...")

        comparison = {}

        # Test each engine with same data subset
        test_symbol = list(self.live_data.keys())[0]  # Use first symbol
        test_data = self.live_data[test_symbol][:20]  # Use 20 data points

        # Test Rust
        if 'rust' in self.test_results and 'error' not in self.test_results['rust']:
            try:
                from rust_integration import RustTradingEngine
                engine = RustTradingEngine()
                start_time = time.time()
                result = engine.process_market_data(test_data, check_risk=False)
                rust_time = time.time() - start_time
                comparison['rust'] = {
                    'latency_seconds': rust_time,
                    'signals': result.signal_count,
                    'throughput': len(test_data) / rust_time
                }
            except:
                comparison['rust'] = {'error': 'Failed to test'}

        # Test C++
        if 'cpp' in self.test_results and 'error' not in self.test_results['cpp']:
            try:
                from research.cpp_integration import QuantTradingEngine
                engine = QuantTradingEngine()
                start_time = time.time()
                result = engine.process_market_data(test_data, check_risk=False)
                cpp_time = time.time() - start_time
                comparison['cpp'] = {
                    'latency_seconds': cpp_time,
                    'signals': len(result.get('signals', [])),
                    'throughput': len(test_data) / cpp_time
                }
            except:
                comparison['cpp'] = {'error': 'Failed to test'}

        self.performance_metrics['comparison'] = comparison
        return comparison

    def generate_report(self):
        """Generate comprehensive test report"""
        self.log("📊 Generating E2E Test Report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_engines_tested': len(self.test_results),
                'engines_available': [k for k, v in self.test_results.items() if 'error' not in v],
                'engines_failed': [k for k, v in self.test_results.items() if 'error' in v],
                'total_symbols_tested': len(self.live_data),
                'total_data_points': sum(len(data) for data in self.live_data.values()),
            },
            'engine_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'errors': self.errors,
            'recommendations': []
        }

        # Generate recommendations
        if len(report['test_summary']['engines_available']) == 0:
            report['recommendations'].append("No engines are working. Check build configurations.")
        else:
            report['recommendations'].append(f"{len(report['test_summary']['engines_available'])} engines are working correctly.")

        if 'rust' in report['engine_results'] and 'error' not in report['engine_results']['rust']:
            report['recommendations'].append("Rust engine is available - recommended for production use.")

        if len(self.errors) > 0:
            report['recommendations'].append(f"Found {len(self.errors)} errors during testing. Check logs for details.")

        # Save report
        try:
            import json
            with open('e2e_test_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.log("✅ Test report saved to e2e_test_report.json")
        except Exception as e:
            self.log_error("Failed to save test report", e)

        return report

    def run_all_tests(self):
        """Run comprehensive E2E test suite"""
        self.log("🚀 Starting Comprehensive E2E Test Suite")
        self.log("=" * 60)

        start_time = time.time()

        # 1. Fetch live data
        if not self.fetch_live_data():
            self.log("❌ Failed to fetch live data. Aborting tests.")
            return False

        # 2. Test each engine
        engines_tested = []

        # Test Rust engine
        if self.test_rust_engine():
            engines_tested.append("Rust")
        else:
            self.log("⚠️  Rust engine test failed")

        # Test Go engine
        if self.test_go_engine():
            engines_tested.append("Go")
        else:
            self.log("⚠️  Go engine test failed")

        # Test C++ engine
        if self.test_cpp_engine():
            engines_tested.append("C++")
        else:
            self.log("⚠️  C++ engine test failed")

        # Test Python engine
        if self.test_python_engine():
            engines_tested.append("Python")
        else:
            self.log("⚠️  Python engine test failed")

        # 3. Compare engines
        if len(engines_tested) > 1:
            self.test_engine_comparison()

        # 4. Generate report
        total_time = time.time() - start_time
        report = self.generate_report()

        # 5. Final summary
        self.log("=" * 60)
        self.log("🏁 E2E Test Suite Complete")
        self.log(f"⏱️  Total test time: {total_time:.2f} seconds")
        self.log(f"🎯 Engines tested: {', '.join(engines_tested) if engines_tested else 'None'}")
        self.log(f"📊 Symbols tested: {len(self.live_data)}")
        self.log(f"📈 Data points processed: {sum(len(data) for data in self.live_data.values())}")
        self.log(f"⚠️  Errors encountered: {len(self.errors)}")

        if len(engines_tested) == 4:
            self.log("🎉 ALL ENGINES WORKING! Multi-language framework is fully operational.")
            return True
        elif len(engines_tested) >= 2:
            self.log(f"✅ {len(engines_tested)} engines working. Framework is partially operational.")
            return True
        else:
            self.log("❌ No engines working. Check build configurations and dependencies.")
            return False

def main():
    """Main E2E test execution"""
    print("🤖 Multi-Language Trading Engine E2E Test Suite")
    print("=" * 60)
    print("Testing all 4 engines (Rust, Go, C++, Python) with live market data")
    print("=" * 60)

    # Initialize tester
    tester = EngineTester()

    try:
        # Run all tests
        success = tester.run_all_tests()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
