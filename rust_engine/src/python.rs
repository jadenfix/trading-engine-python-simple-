//! Python bindings for the Rust trading engine
//!
//! This module provides seamless integration with Python, allowing
//! Python code to use the high-performance Rust trading engine.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for TradingEngine
#[pymodule]
fn rust_trading_engine_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TradingEngine>()?;
    m.add_class::<TradingResult>()?;
    m.add_class::<PerformanceCounters>()?;
    m.add_class::<CacheStats>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ENGINE_NAME", "Rust High-Performance Trading Engine")?;

    Ok(())
}

// Re-export for Python module
pub use crate::lib::*;
