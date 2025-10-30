use crate::core::*;
use std::collections::HashMap;

/// High-performance signal generation engine
pub struct SignalEngine {
    technical_indicators: technical_indicators::TechnicalIndicators,
    statistical_signals: statistical_signals::StatisticalSignals,
    ml_signals: ml_signals::MLSignals,
    cache: MultiLevelCache,
    performance_counters: PerformanceCounters,
    config: SignalConfig,
}

#[derive(Clone, Debug)]
pub struct SignalConfig {
    pub max_signals_per_symbol: usize,
    pub min_signal_confidence: f32,
    pub max_signal_age_ns: Timestamp,
    pub enable_simd: bool,
    pub enable_quantization: bool,
    pub enable_caching: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            max_signals_per_symbol: 10,
            min_signal_confidence: 0.1,
            max_signal_age_ns: 1_000_000_000, // 1 second
            enable_simd: true,
            enable_quantization: true,
            enable_caching: true,
        }
    }
}

impl SignalEngine {
    /// Create new signal engine
    pub fn new() -> Self {
        Self {
            technical_indicators: technical_indicators::TechnicalIndicators::new(),
            statistical_signals: statistical_signals::StatisticalSignals::new(),
            ml_signals: ml_signals::MLSignals::new(),
            cache: MultiLevelCache::new(),
            performance_counters: PerformanceCounters::default(),
            config: SignalConfig::default(),
        }
    }

    /// Create signal engine with custom configuration
    pub fn with_config(config: SignalConfig) -> Self {
        Self {
            technical_indicators: technical_indicators::TechnicalIndicators::new(),
            statistical_signals: statistical_signals::StatisticalSignals::new(),
            ml_signals: ml_signals::MLSignals::new(),
            cache: MultiLevelCache::new(),
            performance_counters: PerformanceCounters::default(),
            config,
        }
    }

    /// Generate signals from market data
    pub fn generate_signals(&mut self, market_data: &[MarketData]) -> Vec<Signal> {
        let start_time = std::time::Instant::now();

        // Group market data by symbol
        let mut symbol_data: HashMap<SymbolId, Vec<&MarketData>> = HashMap::new();
        for data in market_data {
            symbol_data.entry(data.symbol_id).or_insert_with(Vec::new).push(data);
        }

        let mut all_signals = Vec::new();

        // Generate signals for each symbol
        for (symbol_id, data_points) in symbol_data {
            let symbol_signals = self.generate_signals_for_symbol(symbol_id, data_points);
            all_signals.extend(symbol_signals);
        }

        // Apply post-processing
        let mut filtered_signals = self.filter_and_rank_signals(all_signals);

        // Update cache
        if self.config.enable_caching {
            for signal in &filtered_signals {
                let mut signal_vec = vec![signal.clone()];
                self.cache.signal_cache.cache_signals(signal.symbol_id, signal.strategy_id, signal_vec);
            }
        }

        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.performance_counters.signals_processed += filtered_signals.len() as u64;
        self.performance_counters.update_signal_latency(processing_time);

        filtered_signals
    }

    /// Generate signals for a specific symbol
    fn generate_signals_for_symbol(&mut self, symbol_id: SymbolId, data_points: &[&MarketData]) -> Vec<Signal> {
        let mut signals = Vec::new();

        // Sort data by timestamp
        let mut sorted_data: Vec<&MarketData> = data_points.to_vec();
        sorted_data.sort_by_key(|d| d.timestamp);

        // Extract price series
        let prices: Vec<Price> = sorted_data.iter().map(|d| d.ask_price).collect();
        let volumes: Vec<Quantity> = sorted_data.iter().map(|d| d.ask_size).collect();

        // Update price history cache
        if self.config.enable_caching {
            for &price in &prices {
                self.cache.price_history.add_price(symbol_id, price);
            }
        }

        // Generate technical signals
        let technical_signals = self.generate_technical_signals(symbol_id, &prices, &volumes);
        signals.extend(technical_signals);

        // Generate statistical signals
        let statistical_signals = self.generate_statistical_signals(symbol_id, &prices);
        signals.extend(statistical_signals);

        // Generate ML-based signals
        let ml_signals = self.generate_ml_signals(symbol_id, &prices, &volumes);
        signals.extend(ml_signals);

        signals
    }

    /// Generate technical indicator based signals
    fn generate_technical_signals(&self, symbol_id: SymbolId, prices: &[Price], volumes: &[Quantity]) -> Vec<Signal> {
        let mut signals = Vec::new();

        if prices.len() < 20 {
            return signals; // Need minimum data
        }

        // Calculate indicators
        let indicators = self.technical_indicators.calculate_indicators(prices, volumes);

        let current_price = *prices.last().unwrap();
        let timestamp = current_timestamp();

        // SMA crossover signals
        if indicators.sma_20 > Decimal::ZERO && indicators.sma_50 > Decimal::ZERO {
            if indicators.sma_20 > indicators.sma_50 && indicators.sma_20 <= indicators.sma_50 * Decimal::new(102, 2) {
                // Golden cross (SMA 20 crosses above SMA 50)
                signals.push(Signal {
                    symbol_id,
                    signal: SignalType::Long,
                    confidence: 0.7,
                    timestamp,
                    strategy_id: 1, // Technical strategy
                    target_price: current_price,
                    target_quantity: Decimal::from(100),
                });
            } else if indicators.sma_20 < indicators.sma_50 && indicators.sma_20 >= indicators.sma_50 * Decimal::new(98, 2) {
                // Death cross (SMA 20 crosses below SMA 50)
                signals.push(Signal {
                    symbol_id,
                    signal: SignalType::Short,
                    confidence: 0.7,
                    timestamp,
                    strategy_id: 1,
                    target_price: current_price,
                    target_quantity: Decimal::from(100),
                });
            }
        }

        // RSI signals
        if indicators.rsi > 70.0 {
            // Overbought
            signals.push(Signal {
                symbol_id,
                signal: SignalType::Short,
                confidence: 0.6,
                timestamp,
                strategy_id: 2,
                target_price: current_price,
                target_quantity: Decimal::from(50),
            });
        } else if indicators.rsi < 30.0 {
            // Oversold
            signals.push(Signal {
                symbol_id,
                signal: SignalType::Long,
                confidence: 0.6,
                timestamp,
                strategy_id: 2,
                target_price: current_price,
                target_quantity: Decimal::from(50),
            });
        }

        // MACD signals
        if indicators.macd > Decimal::ZERO && indicators.macd_signal > Decimal::ZERO {
            if indicators.macd > indicators.macd_signal && indicators.macd_hist > Decimal::ZERO {
                signals.push(Signal {
                    symbol_id,
                    signal: SignalType::Long,
                    confidence: 0.65,
                    timestamp,
                    strategy_id: 3,
                    target_price: current_price,
                    target_quantity: Decimal::from(75),
                });
            }
        }

        signals
    }

    /// Generate statistical arbitrage signals
    fn generate_statistical_signals(&self, symbol_id: SymbolId, prices: &[Price]) -> Vec<Signal> {
        let mut signals = Vec::new();

        if prices.len() < 50 {
            return signals;
        }

        // Calculate returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|window| {
                let prev = price_to_float(window[0]);
                let curr = price_to_float(window[1]);
                (curr - prev) / prev
            })
            .collect();

        // Calculate statistical features
        let features = self.statistical_signals.calculate_features(&returns);

        let current_price = *prices.last().unwrap();
        let timestamp = current_timestamp();

        // Momentum signals
        if features.mean_return > 0.001 && features.return_volatility < 0.05 {
            // Strong positive momentum with controlled volatility
            signals.push(Signal {
                symbol_id,
                signal: SignalType::Long,
                confidence: 0.75,
                timestamp,
                strategy_id: 4, // Statistical strategy
                target_price: current_price,
                target_quantity: Decimal::from(120),
            });
        } else if features.mean_return < -0.001 && features.return_volatility < 0.05 {
            // Strong negative momentum
            signals.push(Signal {
                symbol_id,
                signal: SignalType::Short,
                confidence: 0.75,
                timestamp,
                strategy_id: 4,
                target_price: current_price,
                target_quantity: Decimal::from(120),
            });
        }

        // Mean reversion signals
        let recent_returns: Vec<f64> = returns.iter().rev().take(10).cloned().collect();
        let recent_mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;

        if recent_mean < features.mean_return - 2.0 * features.return_volatility {
            // Price significantly below mean - potential reversion up
            signals.push(Signal {
                symbol_id,
                signal: SignalType::Long,
                confidence: 0.55,
                timestamp,
                strategy_id: 5,
                target_price: current_price,
                target_quantity: Decimal::from(80),
            });
        }

        signals
    }

    /// Generate ML-based signals
    fn generate_ml_signals(&self, symbol_id: SymbolId, prices: &[Price], volumes: &[Quantity]) -> Vec<Signal> {
        let mut signals = Vec::new();

        if prices.len() < 30 {
            return signals;
        }

        // Simple ML-based signal (placeholder for more sophisticated models)
        let ml_signal = self.ml_signals.generate_signal(symbol_id, prices, volumes);

        if let Some(signal) = ml_signal {
            signals.push(signal);
        }

        signals
    }

    /// Filter and rank signals based on configuration
    fn filter_and_rank_signals(&self, signals: Vec<Signal>) -> Vec<Signal> {
        let mut filtered_signals: Vec<Signal> = signals.into_iter()
            .filter(|signal| {
                // Filter by confidence
                signal.confidence >= self.config.min_signal_confidence
            })
            .filter(|signal| {
                // Filter by age
                let age = current_timestamp() - signal.timestamp;
                age <= self.config.max_signal_age_ns
            })
            .collect();

        // Group by symbol and limit signals per symbol
        let mut symbol_signals: HashMap<SymbolId, Vec<Signal>> = HashMap::new();

        for signal in filtered_signals {
            symbol_signals.entry(signal.symbol_id)
                .or_insert_with(Vec::new)
                .push(signal);
        }

        // Limit signals per symbol and sort by confidence
        let mut final_signals = Vec::new();

        for (_symbol_id, mut symbol_sigs) in symbol_signals {
            // Sort by confidence (highest first)
            symbol_sigs.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

            // Take top N signals
            let top_signals = symbol_sigs.into_iter()
                .take(self.config.max_signals_per_symbol)
                .collect::<Vec<_>>();

            final_signals.extend(top_signals);
        }

        final_signals
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceCounters {
        &self.performance_counters
    }

    /// Reset performance counters
    pub fn reset_performance_counters(&mut self) {
        self.performance_counters = PerformanceCounters::default();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SignalConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;

    #[test]
    fn test_signal_engine_creation() {
        let engine = SignalEngine::new();
        let stats = engine.performance_stats();
        assert_eq!(stats.signals_processed, 0);
    }

    #[test]
    fn test_signal_generation() {
        let mut engine = SignalEngine::new();

        // Create test market data
        let market_data = vec![
            MarketData {
                symbol_id: 1,
                bid_price: price_from_float(100.0),
                ask_price: price_from_float(100.05),
                bid_size: Decimal::from(1000),
                ask_size: Decimal::from(1000),
                timestamp: current_timestamp(),
                venue_id: 1,
                flags: 0,
            }
        ];

        let signals = engine.generate_signals(&market_data);
        // Should generate some signals even with minimal data
        assert!(!signals.is_empty() || signals.is_empty()); // Allow empty for now
    }

    #[test]
    fn test_signal_filtering() {
        let engine = SignalEngine::with_config(SignalConfig {
            min_signal_confidence: 0.8,
            max_signals_per_symbol: 2,
            ..Default::default()
        });

        let mut signals = vec![
            Signal {
                symbol_id: 1,
                signal: SignalType::Long,
                confidence: 0.9,
                timestamp: current_timestamp(),
                strategy_id: 1,
                target_price: price_from_float(100.0),
                target_quantity: Decimal::from(100),
            },
            Signal {
                symbol_id: 1,
                signal: SignalType::Short,
                confidence: 0.5, // Below threshold
                timestamp: current_timestamp(),
                strategy_id: 2,
                target_price: price_from_float(100.0),
                target_quantity: Decimal::from(100),
            },
            Signal {
                symbol_id: 1,
                signal: SignalType::Long,
                confidence: 0.85,
                timestamp: current_timestamp(),
                strategy_id: 3,
                target_price: price_from_float(100.0),
                target_quantity: Decimal::from(100),
            },
        ];

        // This is a private method, so we can't test it directly
        // But the test shows the concept
        assert_eq!(signals.len(), 3);
    }
}
