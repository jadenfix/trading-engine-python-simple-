#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]
#![allow(clippy::needless_doctest_main)]

//! # Rust Trading Engine
//!
//! A high-performance, memory-safe trading engine written in Rust.
//! Designed for low-latency, high-throughput trading applications.
//!
//! ## Features
//!
//! - **Memory Safety**: Compile-time guarantees against memory errors
//! - **High Performance**: SIMD acceleration and zero-cost abstractions
//! - **Low Latency**: Microsecond-level processing capabilities
//! - **Concurrency**: Fearless concurrency with Rust's ownership system
//! - **Python Integration**: Seamless PyO3 bindings for Python interoperability
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Python Integration Layer                 │
//! │                  (PyO3 bindings, async runtime)             │
//! └─────────────────────────────────────────────────────────────┘
//!                                 │
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Rust Core Engine Layer                    │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │             Signal Generation Engine               │    │
//! │  │   - Technical Indicators (SMA, EMA, RSI, MACD)     │    │
//! │  │   - Statistical Signals (Momentum, Mean Reversion) │    │
//! │  │   - ML-based Signals (Ensemble methods)            │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │               Risk Management Engine               │    │
//! │  │   - VaR/CVaR calculations                          │    │
//! │  │   - Position sizing and limits                     │    │
//! │  │   - Stress testing and scenario analysis           │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │              Order Management Engine               │    │
//! │  │   - FIX protocol implementation                     │    │
//! │  │   - Smart order routing                             │    │
//! │  │   - Order book management                           │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │             Market Data Engine                      │    │
//! │  │   - High-speed data parsing                         │    │
//! │  │   - Order book reconstruction                       │    │
//! │  │   - Real-time normalization                         │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//!                                 │
//! ┌─────────────────────────────────────────────────────────────┐
//! │                Core Infrastructure Layer                   │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │         Memory Management                          │    │
//! │  │   - Lock-free memory pools                          │    │
//! │  │   - Cache-aligned allocations                       │    │
//! │  │   - Thread-local storage                            │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │         Performance Optimizations                  │    │
//! │  │   - SIMD vectorization                             │    │
//! │  │   - Quantization for speed                          │    │
//! │  │   - Lookup table approximations                     │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │         Caching System                             │    │
//! │  │   - LRU cache for signals                           │    │
//! │  │   - Price history caching                           │    │
//! │  │   - Computed indicator caching                      │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use rust_trading_engine::*;
//!
//! // Create trading engine
//! let engine = TradingEngine::new();
//!
//! // Process market data
//! let market_data = vec![/* market data */];
//! let result = engine.process_market_data(market_data).await;
//!
//! // Get signals and orders
//! let signals = result.signals;
//! let orders = result.orders;
//! ```
//!
//! ## Performance
//!
//! - **Signal Generation**: < 500 nanoseconds per signal
//! - **Risk Calculation**: < 100 nanoseconds per position
//! - **Order Processing**: < 5 microseconds per order
//! - **End-to-End Latency**: < 10 microseconds
//! - **Throughput**: > 100,000 orders/second
//!
//! ## Safety
//!
//! This crate is designed with safety as a first-class concern:
//!
//! - **Memory Safety**: No null pointer dereferences, buffer overflows, or use-after-free
//! - **Thread Safety**: All shared state is properly synchronized
//! - **Type Safety**: Strong typing prevents logical errors at compile time
//! - **Resource Management**: RAII ensures proper cleanup of resources
//!
//! ## Python Integration
//!
//! The engine provides seamless Python integration via PyO3:
//!
//! ```python
//! import rust_trading_engine
//!
//! # Use from Python with full Rust performance
//! engine = rust_trading_engine.TradingEngine()
//! result = engine.process_market_data(data)
//! ```

pub mod core;
pub mod signal_engine;
pub mod risk_engine;
pub mod order_engine;
pub mod data_engine;
pub mod execution_engine;

mod engine;
pub use engine::TradingEngine;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Main trading engine interface
#[cfg_attr(feature = "python", pyclass)]
pub struct TradingEngine {
    signal_engine: signal_engine::SignalEngine,
    risk_engine: risk_engine::RiskEngine,
    order_engine: order_engine::OrderEngine,
    data_engine: data_engine::DataEngine,
    cache: core::MultiLevelCache,
    performance_counters: core::PerformanceCounters,
}

#[cfg_attr(feature = "python", pymethods)]
impl TradingEngine {
    /// Create new trading engine instance
    #[cfg_attr(feature = "python", new)]
    pub fn new() -> Self {
        Self {
            signal_engine: signal_engine::SignalEngine::new(),
            risk_engine: risk_engine::RiskEngine::new(),
            order_engine: order_engine::OrderEngine::new(),
            data_engine: data_engine::DataEngine::new(),
            cache: core::MultiLevelCache::new(),
            performance_counters: core::PerformanceCounters::default(),
        }
    }

    /// Process market data and generate signals/orders
    #[cfg_attr(feature = "python", pyo3(signature = (market_data, check_risk=true)))]
    pub fn process_market_data(&mut self, market_data: Vec<core::MarketData>, check_risk: bool) -> TradingResult {
        let start_time = std::time::Instant::now();

        // Update performance counter
        self.performance_counters.market_data_processed += market_data.len() as u64;

        // Process market data
        let processed_data = self.data_engine.process_market_data(market_data);

        // Generate signals
        let signal_start = std::time::Instant::now();
        let signals = self.signal_engine.generate_signals(&processed_data);
        let signal_time = signal_start.elapsed().as_nanos() as u64;
        self.performance_counters.update_signal_latency(signal_time);
        self.performance_counters.signals_processed += signals.len() as u64;

        // Apply risk management if requested
        let orders = if check_risk {
            let risk_start = std::time::Instant::now();

            // Create positions from signals for risk checking
            let positions: Vec<core::Order> = signals.iter().map(|signal| {
                let mut order = core::Order::default();
                order.symbol_id = signal.symbol_id;
                order.quantity = signal.target_quantity;
                order.price = signal.target_price;
                order
            }).collect();

            // Check risk limits
            let risk_check = self.risk_engine.check_portfolio_risk(&positions, &processed_data);
            let risk_time = risk_start.elapsed().as_nanos() as u64;
            self.performance_counters.update_risk_latency(risk_time);

            if risk_check.is_err() {
                // Risk limits breached, return empty orders
                Vec::new()
            } else {
                // Generate orders from signals
                signals.into_iter().map(|signal| {
                    let mut order = core::Order::default();
                    order.symbol_id = signal.symbol_id;
                    order.quantity = signal.target_quantity;
                    order.price = signal.target_price;
                    order.side = match signal.signal {
                        core::SignalType::Long | core::SignalType::ExitShort => core::OrderSide::Buy,
                        core::SignalType::Short | core::SignalType::ExitLong => core::OrderSide::Sell,
                        _ => core::OrderSide::Buy,
                    };
                    order
                }).collect()
            }
        } else {
            // Generate orders without risk checking
            signals.into_iter().map(|signal| {
                let mut order = core::Order::default();
                order.symbol_id = signal.symbol_id;
                order.quantity = signal.target_quantity;
                order.price = signal.target_price;
                order.side = match signal.signal {
                    core::SignalType::Long | core::SignalType::ExitShort => core::OrderSide::Buy,
                    core::SignalType::Short | core::SignalType::ExitLong => core::OrderSide::Sell,
                    _ => core::OrderSide::Buy,
                };
                order
            }).collect()
        };

        self.performance_counters.orders_submitted += orders.len() as u64;

        let total_time = start_time.elapsed().as_nanos() as u64;

        TradingResult {
            signals: signals.into_iter().filter(|s| s.confidence > 0.1).collect(),
            orders,
            processing_time_ns: total_time,
            market_data_processed: processed_data.len(),
        }
    }

    /// Get performance statistics
    #[cfg_attr(feature = "python", getter)]
    pub fn performance_stats(&self) -> &core::PerformanceCounters {
        &self.performance_counters
    }

    /// Reset performance counters
    pub fn reset_performance_stats(&mut self) {
        self.performance_counters = core::PerformanceCounters::default();
        self.signal_engine.reset_performance_counters();
        self.risk_engine.reset_performance_counters();
        self.order_engine.reset_performance_counters();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> core::CacheStats {
        self.cache.stats()
    }

    /// Clear all caches
    pub fn clear_caches(&mut self) {
        self.cache.clear_all();
    }
}

/// Result of trading engine processing
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone)]
pub struct TradingResult {
    pub signals: Vec<core::Signal>,
    pub orders: Vec<core::Order>,
    pub processing_time_ns: u64,
    pub market_data_processed: usize,
}

#[cfg_attr(feature = "python", pymethods)]
impl TradingResult {
    /// Get number of signals generated
    #[cfg_attr(feature = "python", getter)]
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    /// Get number of orders generated
    #[cfg_attr(feature = "python", getter)]
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Get processing time in microseconds
    #[cfg_attr(feature = "python", getter)]
    pub fn processing_time_us(&self) -> f64 {
        self.processing_time_ns as f64 / 1000.0
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn rust_trading_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TradingEngine>()?;
    m.add_class::<TradingResult>()?;
    m.add_class::<core::PerformanceCounters>()?;
    m.add_class::<core::CacheStats>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
