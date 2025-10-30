#!/usr/bin/env python3
"""
Rust Trading Engine Integration

This module provides seamless integration with the high-performance Rust trading engine.
It offers both direct Rust bindings and a Python wrapper with additional functionality.
"""

import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Try to import Rust engine, fallback to Python implementation
try:
    import rust_trading_engine
    RUST_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Rust trading engine loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️  Rust trading engine not available: {e}")
    logger.warning("Falling back to Python implementation")

    # Import fallback Python implementation
    from research.cpp_integration import CitadelTradingEngine as PythonEngine

@dataclass
class MarketData:
    """Market data structure compatible with Rust engine"""
    symbol_id: int
    bid_price: str  # Decimal string for precision
    ask_price: str
    bid_size: str
    ask_size: str
    timestamp: int
    venue_id: int
    flags: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        return cls(
            symbol_id=data.get('symbol_id', 0),
            bid_price=str(data.get('bid_price', '0.0')),
            ask_price=str(data.get('ask_price', '0.0')),
            bid_size=str(data.get('bid_size', '0')),
            ask_size=str(data.get('ask_size', '0')),
            timestamp=data.get('timestamp', 0),
            venue_id=data.get('venue_id', 0),
            flags=data.get('flags', 0),
        )

@dataclass
class Signal:
    """Trading signal structure"""
    symbol_id: int
    signal: str  # 'Long', 'Short', 'Neutral', 'ExitLong', 'ExitShort'
    confidence: float
    timestamp: int
    strategy_id: int
    target_price: str
    target_quantity: str

@dataclass
class Order:
    """Order structure"""
    symbol_id: int
    order_type: str  # 'Market', 'Limit', etc.
    side: str  # 'Buy', 'Sell'
    price: str
    quantity: str
    filled_quantity: str = "0"
    status: str = "Pending"
    create_time: int = 0
    update_time: int = 0
    venue_id: int = 0
    account_id: int = 0
    time_in_force: str = "Day"
    flags: int = 0

@dataclass
class TradingResult:
    """Result from trading engine processing"""
    signals: List[Signal]
    orders: List[Order]
    processing_time_ns: int
    market_data_processed: int

    @property
    def processing_time_us(self) -> float:
        return self.processing_time_ns / 1000.0

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def order_count(self) -> int:
        return len(self.orders)

class RustTradingEngine:
    """
    High-performance Rust trading engine with Python interface.

    This class provides a unified interface that automatically uses the Rust engine
    when available, with graceful fallback to Python implementation.
    """

    def __init__(self):
        """Initialize the trading engine"""
        if RUST_AVAILABLE:
            self._engine = rust_trading_engine.TradingEngine()
            self._engine_type = "rust"
            logger.info("🚀 Using high-performance Rust trading engine")
        else:
            self._engine = PythonEngine()
            self._engine_type = "python"
            logger.info("🐍 Using Python fallback trading engine")

    def process_market_data(
        self,
        market_data: Union[List[Dict], List[MarketData]],
        check_risk: bool = True
    ) -> TradingResult:
        """
        Process market data and generate trading signals/orders.

        Args:
            market_data: List of market data dictionaries or MarketData objects
            check_risk: Whether to perform risk management checks

        Returns:
            TradingResult with signals and orders
        """
        # Convert to Rust-compatible format
        if isinstance(market_data[0], dict):
            rust_data = [MarketData.from_dict(d) for d in market_data]
        else:
            rust_data = market_data

        if self._engine_type == "rust":
            # Convert to Rust format
            rust_market_data = []
            for data in rust_data:
                rust_market_data.append({
                    'symbol_id': data.symbol_id,
                    'bid_price': data.bid_price,
                    'ask_price': data.ask_price,
                    'bid_size': data.bid_size,
                    'ask_size': data.ask_size,
                    'timestamp': data.timestamp,
                    'venue_id': data.venue_id,
                    'flags': data.flags,
                })

            # Process with Rust engine
            result = self._engine.process_market_data(rust_market_data, check_risk)

            # Convert back to Python objects
            signals = []
            for sig in result.signals:
                signals.append(Signal(
                    symbol_id=sig.symbol_id,
                    signal=self._rust_signal_to_string(sig.signal),
                    confidence=sig.confidence,
                    timestamp=sig.timestamp,
                    strategy_id=sig.strategy_id,
                    target_price=str(sig.target_price),
                    target_quantity=str(sig.target_quantity),
                ))

            orders = []
            for order in result.orders:
                orders.append(Order(
                    symbol_id=order.symbol_id,
                    order_type=self._rust_order_type_to_string(order.order_type),
                    side=self._rust_side_to_string(order.side),
                    price=str(order.price),
                    quantity=str(order.quantity),
                    filled_quantity=str(order.filled_quantity),
                    status=self._rust_status_to_string(order.status),
                    create_time=order.create_time,
                    update_time=order.update_time,
                    venue_id=order.venue_id,
                    account_id=order.account_id,
                ))

            return TradingResult(
                signals=signals,
                orders=orders,
                processing_time_ns=result.processing_time_ns,
                market_data_processed=result.market_data_processed,
            )

        else:
            # Use Python fallback
            result = self._engine.process_market_data(market_data, check_risk)

            # Convert to unified format
            signals = [Signal(**sig.__dict__) for sig in result.signals] if hasattr(result, 'signals') else []
            orders = [Order(**order.__dict__) for order in result.orders] if hasattr(result, 'orders') else []

            return TradingResult(
                signals=signals,
                orders=orders,
                processing_time_ns=getattr(result, 'processing_time_ns', 0),
                market_data_processed=len(market_data),
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if RUST_AVAILABLE:
            stats = self._engine.performance_stats()
            return {
                'signals_processed': stats.signals_processed,
                'orders_submitted': stats.orders_submitted,
                'orders_executed': stats.orders_executed,
                'risk_checks_performed': stats.risk_checks_performed,
                'market_data_processed': stats.market_data_processed,
                'avg_signal_latency_ns': stats.avg_signal_latency,
                'avg_order_latency_ns': stats.avg_order_latency,
                'avg_risk_latency_ns': stats.avg_risk_latency,
            }
        else:
            return {
                'engine_type': 'python_fallback',
                'performance': 'degraded',
            }

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        if RUST_AVAILABLE:
            stats = self._engine.cache_stats()
            return {
                'signal_cache_size': stats.signal_cache_size,
                'price_history_symbols': stats.price_history_symbols,
                'indicator_cache_size': stats.indicator_cache_size,
            }
        else:
            return {}

    def reset_performance_stats(self):
        """Reset performance counters"""
        if RUST_AVAILABLE:
            self._engine.reset_performance_stats()

    def clear_caches(self):
        """Clear all caches"""
        if RUST_AVAILABLE:
            self._engine.clear_caches()

    @property
    def engine_type(self) -> str:
        """Get the engine type being used"""
        return self._engine_type

    @property
    def rust_available(self) -> bool:
        """Check if Rust engine is available"""
        return RUST_AVAILABLE

    def _rust_signal_to_string(self, signal_code: int) -> str:
        """Convert Rust signal code to string"""
        signal_map = {
            1: 'Long',
            -1: 'Short',
            0: 'Neutral',
            -2: 'ExitLong',
            2: 'ExitShort',
        }
        return signal_map.get(signal_code, 'Neutral')

    def _rust_order_type_to_string(self, order_type: int) -> str:
        """Convert Rust order type code to string"""
        type_map = {
            0: 'Market',
            1: 'Limit',
            2: 'Stop',
            3: 'StopLimit',
            4: 'TrailingStop',
        }
        return type_map.get(order_type, 'Market')

    def _rust_side_to_string(self, side: int) -> str:
        """Convert Rust side code to string"""
        side_map = {
            0: 'Buy',
            1: 'Sell',
        }
        return side_map.get(side, 'Buy')

    def _rust_status_to_string(self, status: int) -> str:
        """Convert Rust status code to string"""
        status_map = {
            0: 'Pending',
            1: 'PartialFill',
            2: 'Filled',
            3: 'Cancelled',
            4: 'Rejected',
            5: 'Expired',
        }
        return status_map.get(status, 'Pending')

def create_engine() -> RustTradingEngine:
    """Factory function to create trading engine"""
    return RustTradingEngine()

def benchmark_engines():
    """Benchmark Rust vs Python engine performance"""
    import time

    # Create test data
    market_data = []
    for i in range(1000):
        market_data.append({
            'symbol_id': i % 100,
            'bid_price': f"{100.0 + (i % 50)}",
            'ask_price': f"{100.05 + (i % 50)}",
            'bid_size': '1000',
            'ask_size': '1000',
            'timestamp': 1234567890000000000 + i * 1000000,
            'venue_id': 1,
            'flags': 0,
        })

    # Benchmark Rust engine
    rust_times = []
    if RUST_AVAILABLE:
        engine = RustTradingEngine()
        for _ in range(10):
            start = time.perf_counter()
            result = engine.process_market_data(market_data.copy(), check_risk=False)
            end = time.perf_counter()
            rust_times.append((end - start) * 1_000_000)  # microseconds

        rust_avg = sum(rust_times) / len(rust_times)
        print(".2f"
              ".2f"
              f"Signals: {result.signal_count}, Orders: {result.order_count}")

    # Benchmark Python engine (if available)
    try:
        from research.cpp_integration import CitadelTradingEngine
        python_times = []
        python_engine = CitadelTradingEngine()

        for _ in range(10):
            start = time.perf_counter()
            result = python_engine.process_market_data(market_data.copy(), check_risk=False)
            end = time.perf_counter()
            python_times.append((end - start) * 1_000_000)

        python_avg = sum(python_times) / len(python_times)
        print(".2f"
              ".2f")

        if RUST_AVAILABLE:
            speedup = python_avg / rust_avg
            print(".1f")

    except ImportError:
        print("Python engine benchmark not available")

if __name__ == "__main__":
    # Run benchmark when executed directly
    print("🚀 Rust Trading Engine Benchmark")
    print("=" * 50)
    benchmark_engines()
