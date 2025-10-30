use crate::core::*;
use statrs::statistics::Statistics;

/// Statistical signal generation
pub struct StatisticalSignals;

impl StatisticalSignals {
    /// Create new statistical signals generator
    pub fn new() -> Self {
        Self
    }

    /// Calculate statistical features from returns
    pub fn calculate_features(&self, returns: &[f64]) -> StatisticalFeatures {
        if returns.is_empty() {
            return StatisticalFeatures::default();
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        StatisticalFeatures {
            mean_return,
            return_volatility: std_dev,
            skewness: self.calculate_skewness(returns, mean_return, std_dev),
            kurtosis: self.calculate_kurtosis(returns, mean_return, std_dev),
        }
    }

    fn calculate_skewness(&self, returns: &[f64], mean: f64, std: f64) -> f64 {
        if std == 0.0 || returns.len() < 2 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let skewness = returns.iter()
            .map(|r| ((r - mean) / std).powi(3))
            .sum::<f64>() / n;

        skewness
    }

    fn calculate_kurtosis(&self, returns: &[f64], mean: f64, std: f64) -> f64 {
        if std == 0.0 || returns.len() < 2 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let kurtosis = returns.iter()
            .map(|r| ((r - mean) / std).powi(4))
            .sum::<f64>() / n;

        kurtosis - 3.0 // Excess kurtosis
    }
}

#[derive(Clone, Debug)]
pub struct StatisticalFeatures {
    pub mean_return: f64,
    pub return_volatility: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl Default for StatisticalFeatures {
    fn default() -> Self {
        Self {
            mean_return: 0.0,
            return_volatility: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}
