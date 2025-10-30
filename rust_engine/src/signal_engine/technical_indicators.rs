use crate::core::{types::*, quantization};
use rust_decimal::Decimal;

/// Technical indicators calculator with high performance
pub struct TechnicalIndicators {
    quantizer: quantization::PriceQuantizer,
}

#[derive(Clone, Debug)]
pub struct IndicatorValues {
    pub sma_20: f64,
    pub sma_50: f64,
    pub ema_12: f64,
    pub ema_26: f64,
    pub rsi: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub bb_upper: f64,
    pub bb_lower: f64,
    pub bb_middle: f64,
    pub atr: f64,
    pub stoch_k: f64,
    pub stoch_d: f64,
}

impl Default for IndicatorValues {
    fn default() -> Self {
        Self {
            sma_20: 0.0,
            sma_50: 0.0,
            ema_12: 0.0,
            ema_26: 0.0,
            rsi: 50.0,
            macd: 0.0,
            macd_signal: 0.0,
            macd_hist: 0.0,
            bb_upper: 0.0,
            bb_lower: 0.0,
            bb_middle: 0.0,
            atr: 0.0,
            stoch_k: 50.0,
            stoch_d: 50.0,
        }
    }
}

impl TechnicalIndicators {
    /// Create new technical indicators calculator
    pub fn new() -> Self {
        Self {
            quantizer: quantization::PriceQuantizer::default(),
        }
    }

    /// Calculate all indicators for given price and volume data
    pub fn calculate_indicators(&self, prices: &[Price], volumes: &[Quantity]) -> IndicatorValues {
        if prices.is_empty() {
            return IndicatorValues::default();
        }

        let mut indicators = IndicatorValues::default();

        // Convert prices to floats for calculation
        let price_floats: Vec<f64> = prices.iter().map(|p| price_to_float(*p)).collect();

        // Simple Moving Averages
        indicators.sma_20 = self.calculate_sma(&price_floats, 20);
        indicators.sma_50 = self.calculate_sma(&price_floats, 50);

        // Exponential Moving Averages
        indicators.ema_12 = self.calculate_ema(&price_floats, 12);
        indicators.ema_26 = self.calculate_ema(&price_floats, 26);

        // RSI
        indicators.rsi = self.calculate_rsi(&price_floats, 14);

        // MACD
        let (macd, signal, hist) = self.calculate_macd(&price_floats, 12, 26, 9);
        indicators.macd = macd;
        indicators.macd_signal = signal;
        indicators.macd_hist = hist;

        // Bollinger Bands
        let (upper, lower, middle) = self.calculate_bollinger_bands(&price_floats, 20, 2.0);
        indicators.bb_upper = upper;
        indicators.bb_lower = lower;
        indicators.bb_middle = middle;

        // ATR
        indicators.atr = self.calculate_atr(prices, 14);

        // Stochastic Oscillator
        let (stoch_k, stoch_d) = self.calculate_stochastic(prices, 14, 3);
        indicators.stoch_k = stoch_k;
        indicators.stoch_d = stoch_d;

        indicators
    }

    /// Calculate Simple Moving Average
    pub fn calculate_sma(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }

        let sum: f64 = prices.iter().rev().take(period).sum();
        sum / period as f64
    }

    /// Calculate Exponential Moving Average
    pub fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for &price in prices.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }

        ema
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        // Calculate price changes
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        // Calculate average gains and losses
        let avg_gain = self.calculate_sma(&gains, period);
        let avg_loss = self.calculate_sma(&losses, period);

        if avg_loss == 0.0 {
            return 100.0; // Extremely overbought
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn calculate_macd(&self, prices: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> (f64, f64, f64) {
        if prices.len() < slow_period {
            return (0.0, 0.0, 0.0);
        }

        let ema_fast = self.calculate_ema(prices, fast_period);
        let ema_slow = self.calculate_ema(prices, slow_period);

        let macd = ema_fast - ema_slow;
        let signal = self.calculate_ema(&vec![macd; prices.len()], signal_period);
        let histogram = macd - signal;

        (macd, signal, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn calculate_bollinger_bands(&self, prices: &[f64], period: usize, std_dev_multiplier: f64) -> (f64, f64, f64) {
        if prices.len() < period {
            return (0.0, 0.0, 0.0);
        }

        let sma = self.calculate_sma(prices, period);

        // Calculate standard deviation
        let variance = prices.iter().rev().take(period)
            .map(|price| (price - sma).powi(2))
            .sum::<f64>() / period as f64;

        let std_dev = variance.sqrt();

        let upper = sma + std_dev_multiplier * std_dev;
        let lower = sma - std_dev_multiplier * std_dev;

        (upper, lower, sma)
    }

    /// Calculate ATR (Average True Range)
    pub fn calculate_atr(&self, prices: &[Price], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.0;
        }

        let mut true_ranges = Vec::new();

        for i in 1..prices.len() {
            let high = price_to_float(prices[i]);
            let low = price_to_float(prices[i]);
            let prev_close = price_to_float(prices[i - 1]);

            let tr1 = high - low;
            let tr2 = (high - prev_close).abs();
            let tr3 = (low - prev_close).abs();

            let true_range = tr1.max(tr2).max(tr3);
            true_ranges.push(true_range);
        }

        self.calculate_sma(&true_ranges, period)
    }

    /// Calculate Stochastic Oscillator
    pub fn calculate_stochastic(&self, prices: &[Price], k_period: usize, d_period: usize) -> (f64, f64) {
        if prices.len() < k_period {
            return (50.0, 50.0);
        }

        let price_floats: Vec<f64> = prices.iter().map(|p| price_to_float(*p)).collect();

        // Calculate %K
        let mut k_values = Vec::new();

        for i in (k_period - 1)..price_floats.len() {
            let window = &price_floats[i - k_period + 1..=i];
            let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let current = price_floats[i];

            let k = if highest != lowest {
                100.0 * (current - lowest) / (highest - lowest)
            } else {
                50.0
            };

            k_values.push(k);
        }

        // Calculate %D (SMA of %K)
        let d = self.calculate_sma(&k_values, d_period);

        // Return most recent %K and %D
        let k = k_values.last().copied().unwrap_or(50.0);
        (k, d)
    }

    /// Calculate ROC (Rate of Change)
    pub fn calculate_roc(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.0;
        }

        let current = prices[prices.len() - 1];
        let past = prices[prices.len() - 1 - period];

        if past == 0.0 {
            return 0.0;
        }

        ((current - past) / past) * 100.0
    }

    /// Calculate Williams %R
    pub fn calculate_williams_r(&self, prices: &[Price], period: usize) -> f64 {
        if prices.len() < period {
            return -50.0;
        }

        let price_floats: Vec<f64> = prices.iter().map(|p| price_to_float(*p)).collect();
        let window = &price_floats[price_floats.len() - period..];

        let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = price_floats[price_floats.len() - 1];

        if highest == lowest {
            return -50.0;
        }

        -100.0 * (highest - current) / (highest - lowest)
    }

    /// Calculate Commodity Channel Index (CCI)
    pub fn calculate_cci(&self, prices: &[Price], period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }

        let price_floats: Vec<f64> = prices.iter().map(|p| price_to_float(*p)).collect();

        // Calculate Typical Price (SMA of High, Low, Close)
        // Since we only have close prices, use that as approximation
        let tp_values: Vec<f64> = price_floats.iter().rev().take(period).cloned().collect();

        let sma_tp = self.calculate_sma(&tp_values, period);

        // Calculate Mean Deviation
        let mean_deviation: f64 = tp_values.iter()
            .map(|tp| (tp - sma_tp).abs())
            .sum::<f64>() / period as f64;

        if mean_deviation == 0.0 {
            return 0.0;
        }

        let current_tp = tp_values[0]; // Most recent
        (current_tp - sma_tp) / (0.015 * mean_deviation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;

    #[test]
    fn test_sma_calculation() {
        let ti = TechnicalIndicators::new();
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];

        let sma = ti.calculate_sma(&prices, 3);
        assert!((sma - 103.0).abs() < 0.001); // (102+103+104)/3 = 103
    }

    #[test]
    fn test_rsi_calculation() {
        let ti = TechnicalIndicators::new();
        // Create a series with consistent gains
        let prices = (0..20).map(|i| 100.0 + i as f64).collect::<Vec<f64>>();

        let rsi = ti.calculate_rsi(&prices, 14);
        assert!(rsi > 50.0); // Should be high due to consistent gains
    }

    #[test]
    fn test_bollinger_bands() {
        let ti = TechnicalIndicators::new();
        let prices = vec![100.0; 20]; // Constant prices

        let (upper, lower, middle) = ti.calculate_bollinger_bands(&prices, 20, 2.0);
        assert_eq!(middle, 100.0);
        assert!(upper > middle);
        assert!(lower < middle);
        assert_eq!(upper - middle, middle - lower); // Symmetric
    }

    #[test]
    fn test_stochastic() {
        let ti = TechnicalIndicators::new();
        // Create increasing prices
        let prices: Vec<Price> = (100..120).map(|p| price_from_float(p as f64)).collect();

        let (stoch_k, stoch_d) = ti.calculate_stochastic(&prices, 14, 3);
        assert!(stoch_k >= 0.0 && stoch_k <= 100.0);
        assert!(stoch_d >= 0.0 && stoch_d <= 100.0);
    }
}
