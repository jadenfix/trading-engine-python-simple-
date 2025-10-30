use crate::core::*;

/// Machine learning based signal generation
pub struct MLSignals;

impl MLSignals {
    /// Create new ML signals generator
    pub fn new() -> Self {
        Self
    }

    /// Generate ML-based signal (simplified placeholder)
    pub fn generate_signal(&self, symbol_id: SymbolId, prices: &[Price], volumes: &[Quantity]) -> Option<Signal> {
        if prices.len() < 10 {
            return None;
        }

        // Simple momentum-based signal as placeholder for ML
        let recent_prices: Vec<f64> = prices.iter().rev().take(5).map(|p| price_to_float(*p)).collect();
        let older_prices: Vec<f64> = prices.iter().rev().skip(5).take(5).map(|p| price_to_float(*p)).collect();

        if recent_prices.is_empty() || older_prices.is_empty() {
            return None;
        }

        let recent_avg = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let older_avg = older_prices.iter().sum::<f64>() / older_prices.len() as f64;

        let momentum = (recent_avg - older_avg) / older_avg;

        let (signal_type, confidence) = if momentum > 0.02 {
            (SignalType::Long, 0.8)
        } else if momentum < -0.02 {
            (SignalType::Short, 0.8)
        } else {
            (SignalType::Neutral, 0.5)
        };

        Some(Signal {
            symbol_id,
            signal: signal_type,
            confidence,
            timestamp: current_timestamp(),
            strategy_id: 100, // ML strategy
            target_price: *prices.last().unwrap(),
            target_quantity: Decimal::from(50),
        })
    }
}
