use crate::core::*;

/// Value at Risk (VaR) and CVaR calculator
pub struct VaRCalculator;

impl VaRCalculator {
    /// Create new VaR calculator
    pub fn new() -> Self {
        Self
    }

    /// Calculate historical VaR
    pub fn calculate_historical_var(&self, returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let index = index.min(sorted_returns.len() - 1);

        -sorted_returns[index] // Negative because VaR is loss
    }

    /// Calculate Conditional VaR (CVaR/Expected Shortfall)
    pub fn calculate_cvar(&self, returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let tail_returns: Vec<f64> = sorted_returns[..=index].iter().map(|&x| -x).collect();

        if tail_returns.is_empty() {
            0.0
        } else {
            tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        }
    }

    /// Calculate parametric VaR (assuming normal distribution)
    pub fn calculate_parametric_var(&self, returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        // Use normal distribution quantile
        let z_score = match confidence_level {
            0.95 => 1.645,
            0.99 => 2.326,
            0.999 => 3.090,
            _ => 1.645, // Default to 95%
        };

        -(mean - z_score * std_dev)
    }
}
