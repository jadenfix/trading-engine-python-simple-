use crate::core::*;
use std::collections::HashMap;

/// High-performance market data processing engine
pub struct DataEngine {
    data_cache: HashMap<SymbolId, Vec<MarketData>>,
    max_history: usize,
}

impl DataEngine {
    /// Create new data engine
    pub fn new() -> Self {
        Self {
            data_cache: HashMap::new(),
            max_history: 1000,
        }
    }

    /// Process incoming market data
    pub fn process_market_data(&mut self, market_data: Vec<MarketData>) -> Vec<MarketData> {
        for data in &market_data {
            let history = self.data_cache.entry(data.symbol_id).or_insert_with(Vec::new);
            history.push(data.clone());

            // Keep only recent data
            if history.len() > self.max_history {
                history.remove(0);
            }
        }

        market_data
    }

    /// Get historical data for symbol
    pub fn get_historical_data(&self, symbol_id: SymbolId, n_points: usize) -> Vec<&MarketData> {
        if let Some(history) = self.data_cache.get(&symbol_id) {
            let start = history.len().saturating_sub(n_points);
            history[start..].iter().collect()
        } else {
            Vec::new()
        }
    }

    /// Get latest market data for symbol
    pub fn get_latest_data(&self, symbol_id: SymbolId) -> Option<&MarketData> {
        self.data_cache.get(&symbol_id)?.last()
    }

    /// Clear data cache
    pub fn clear_cache(&mut self) {
        self.data_cache.clear();
    }
}
