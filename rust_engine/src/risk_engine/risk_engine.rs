use crate::core::*;
use std::collections::HashMap;

/// High-performance risk management engine
pub struct RiskEngine {
    var_calculator: var_calculator::VaRCalculator,
    position_manager: position_manager::PositionManager,
    stress_tester: stress_tester::StressTester,
    risk_limits: RiskLimits,
    cache: HashMap<SymbolId, RiskMetrics>,
    performance_counters: PerformanceCounters,
}

#[derive(Clone, Debug)]
pub struct RiskLimits {
    pub max_portfolio_var: f64,
    pub max_position_var: f64,
    pub max_drawdown: f64,
    pub max_leverage: f64,
    pub max_concentration: f64,
    pub max_positions: usize,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_portfolio_var: 0.05,  // 5% VaR limit
            max_position_var: 0.02,   // 2% per position VaR
            max_drawdown: 0.10,       // 10% max drawdown
            max_leverage: 5.0,        // 5x max leverage
            max_concentration: 0.25,  // 25% max position size
            max_positions: 100,       // Max number of positions
        }
    }
}

impl RiskEngine {
    /// Create new risk engine
    pub fn new() -> Self {
        Self {
            var_calculator: var_calculator::VaRCalculator::new(),
            position_manager: position_manager::PositionManager::new(),
            stress_tester: stress_tester::StressTester::new(),
            risk_limits: RiskLimits::default(),
            cache: HashMap::new(),
            performance_counters: PerformanceCounters::default(),
        }
    }

    /// Calculate portfolio-level risk metrics
    pub fn calculate_portfolio_risk(&mut self, positions: &[Order], market_data: &[MarketData]) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Group market data by symbol
        let mut symbol_data: HashMap<SymbolId, &MarketData> = HashMap::new();
        for data in market_data {
            symbol_data.insert(data.symbol_id, data);
        }

        // Calculate risk for each position
        let mut total_exposure = Decimal::ZERO;
        let mut position_risks = Vec::new();

        for position in positions {
            if let Some(market_data) = symbol_data.get(&position.symbol_id) {
                let risk_metrics = self.calculate_position_risk(position, market_data)?;
                position_risks.push(risk_metrics);
                total_exposure += position.quantity * market_data.ask_price;
            }
        }

        // Check risk limits
        self.check_portfolio_limits(&position_risks, total_exposure)?;

        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.performance_counters.update_risk_latency(processing_time);
        self.performance_counters.risk_checks_performed += 1;

        Ok(())
    }

    /// Calculate risk metrics for a single position
    pub fn calculate_position_risk(&mut self, position: &Order, market_data: &MarketData) -> Result<RiskMetrics> {
        let current_price = market_data.ask_price;

        // Check cache first
        if let Some(cached_risk) = self.cache.get(&position.symbol_id) {
            let mut risk_metrics = cached_risk.clone();
            risk_metrics.update_position(current_price);
            return Ok(risk_metrics);
        }

        // Calculate VaR using historical simulation (simplified)
        let returns = self.generate_synthetic_returns(252); // 1 year of daily returns
        let var_95 = self.var_calculator.calculate_historical_var(&returns, 0.95);
        let expected_shortfall = self.var_calculator.calculate_cvar(&returns, 0.95);

        // Calculate Sharpe ratio (simplified)
        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let sharpe_ratio = if var_95 > 0.0 {
            avg_return / var_95
        } else {
            0.0
        };

        let mut risk_metrics = RiskMetrics {
            symbol_id: position.symbol_id,
            current_price,
            position_size: position.quantity,
            entry_price: position.price,
            stop_loss_price: position.price * Decimal::new(95, 2), // 5% stop loss
            take_profit_price: position.price * Decimal::new(110, 2), // 10% take profit
            var_95: var_95.abs(),
            expected_shortfall: expected_shortfall.abs(),
            sharpe_ratio,
            max_drawdown: 0.05, // Placeholder
            timestamp: current_timestamp(),
            unrealized_pnl: Decimal::ZERO,
            position_var: 0.0,
            breach_warning: false,
        };

        risk_metrics.update_position(current_price);

        // Cache the result
        self.cache.insert(position.symbol_id, risk_metrics.clone());

        Ok(risk_metrics)
    }

    /// Check if portfolio risk limits are breached
    pub fn check_portfolio_limits(&self, position_risks: &[RiskMetrics], total_exposure: Decimal) -> Result<()> {
        let total_var: f64 = position_risks.iter().map(|r| r.position_var).sum();

        // Check VaR limit
        if total_var > self.risk_limits.max_portfolio_var {
            return Err(TradingError::RiskLimitBreached {
                limit: format!("Portfolio VaR {:.4} exceeds limit {:.4}", total_var, self.risk_limits.max_portfolio_var)
            });
        }

        // Check position concentration
        for risk in position_risks {
            let position_exposure = risk.position_size * risk.current_price;
            let concentration = price_to_float(position_exposure) / price_to_float(total_exposure);

            if concentration > self.risk_limits.max_concentration {
                return Err(TradingError::RiskLimitBreached {
                    limit: format!("Position concentration {:.2}% exceeds limit {:.2}%",
                                 concentration * 100.0, self.risk_limits.max_concentration * 100.0)
                });
            }

            // Check individual position VaR
            if risk.position_var > self.risk_limits.max_position_var {
                return Err(TradingError::RiskLimitBreached {
                    limit: format!("Position VaR {:.4} exceeds limit {:.4}", risk.position_var, self.risk_limits.max_position_var)
                });
            }
        }

        Ok(())
    }

    /// Run stress tests on portfolio
    pub fn run_stress_tests(&self, positions: &[Order], market_data: &[MarketData]) -> Vec<RiskMetrics> {
        self.stress_tester.run_stress_tests(positions, market_data)
    }

    /// Update risk limits
    pub fn update_risk_limits(&mut self, limits: RiskLimits) {
        self.risk_limits = limits;
    }

    /// Get current risk limits
    pub fn get_risk_limits(&self) -> &RiskLimits {
        &self.risk_limits
    }

    /// Calculate Kelly criterion for position sizing
    pub fn calculate_kelly_criterion(&self, returns: &[f64], win_rate: Option<f64>, win_loss_ratio: Option<f64>) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        // Method 1: Using win rate and win/loss ratio (if provided)
        if let (Some(win_rate), Some(win_loss_ratio)) = (win_rate, win_loss_ratio) {
            let loss_rate = 1.0 - win_rate;
            return win_rate / win_loss_ratio - loss_rate;
        }

        // Method 2: Using historical returns
        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        // Kelly fraction = (expected return) / variance
        avg_return / variance
    }

    /// Calculate maximum drawdown
    pub fn calculate_max_drawdown(&self, portfolio_values: &[f64]) -> f64 {
        if portfolio_values.is_empty() {
            return 0.0;
        }

        let mut max_drawdown = 0.0;
        let mut peak = portfolio_values[0];

        for &value in portfolio_values {
            if value > peak {
                peak = value;
            }

            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Generate synthetic returns for VaR calculation
    fn generate_synthetic_returns(&self, n_returns: usize) -> Vec<f64> {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0001, 0.02).unwrap(); // Mean return 0.01%, std 2%

        (0..n_returns)
            .map(|_| normal.sample(&mut rng))
            .collect()
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceCounters {
        &self.performance_counters
    }

    /// Reset performance counters
    pub fn reset_performance_counters(&mut self) {
        self.performance_counters = PerformanceCounters::default();
    }

    /// Clear risk cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;

    #[test]
    fn test_risk_engine_creation() {
        let engine = RiskEngine::new();
        let stats = engine.performance_stats();
        assert_eq!(stats.risk_checks_performed, 0);
    }

    #[test]
    fn test_position_risk_calculation() {
        let mut engine = RiskEngine::new();

        let order = Order {
            symbol_id: 1,
            quantity: Decimal::from(100),
            price: price_from_float(100.0),
            ..Default::default()
        };

        let market_data = MarketData {
            symbol_id: 1,
            bid_price: price_from_float(101.0),
            ask_price: price_from_float(101.0),
            bid_size: Decimal::from(1000),
            ask_size: Decimal::from(1000),
            timestamp: current_timestamp(),
            venue_id: 1,
            flags: 0,
        };

        let risk = engine.calculate_position_risk(&order, &market_data).unwrap();
        assert_eq!(risk.symbol_id, 1);
        assert!(risk.var_95 >= 0.0);
        assert!(risk.expected_shortfall >= 0.0);
    }

    #[test]
    fn test_kelly_criterion() {
        let engine = RiskEngine::new();

        // Test with historical returns
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];
        let kelly = engine.calculate_kelly_criterion(&returns, None, None);
        assert!(kelly.is_finite());

        // Test with win rate parameters
        let kelly_params = engine.calculate_kelly_criterion(&[], Some(0.6), Some(2.0));
        assert!((kelly_params - 0.1).abs() < 0.001); // (0.6/2.0) - 0.4 = 0.1
    }

    #[test]
    fn test_max_drawdown() {
        let engine = RiskEngine::new();

        let portfolio_values = vec![100.0, 95.0, 98.0, 90.0, 105.0, 95.0];
        let max_dd = engine.calculate_max_drawdown(&portfolio_values);

        // Expected: (100 - 90) / 100 = 0.1
        assert!((max_dd - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_risk_limits() {
        let engine = RiskEngine::new();
        let limits = engine.get_risk_limits();

        assert_eq!(limits.max_portfolio_var, 0.05);
        assert_eq!(limits.max_drawdown, 0.10);
        assert_eq!(limits.max_positions, 100);
    }
}
