#pragma once

#include "../core/types.hpp"
#include "../core/quantization.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace trading {

// Forward declarations
class VaRCalculator;
class PositionManager;
class StressTester;

class RiskEngine {
public:
    RiskEngine();
    ~RiskEngine() = default;

    // Disable copy/move for performance
    RiskEngine(const RiskEngine&) = delete;
    RiskEngine& operator=(const RiskEngine&) = delete;
    RiskEngine(RiskEngine&&) = delete;
    RiskEngine& operator=(RiskEngine&&) = delete;

    // Core risk assessment
    RiskMetrics calculate_portfolio_risk(const std::vector<Order>& positions,
                                       const std::vector<MarketData>& market_data);

    bool check_risk_limits(const std::vector<Order>& positions,
                          const RiskMetrics& current_risk) noexcept;

    // Position-level risk calculations
    RiskMetrics calculate_position_risk(const Order& position,
                                      const MarketData& market_data) noexcept;

    // Portfolio-level risk calculations
    RiskMetrics calculate_portfolio_risk_metrics(const std::vector<Order>& positions,
                                                const std::vector<MarketData>& market_data);

    // Real-time risk monitoring
    void update_risk_metrics(std::vector<RiskMetrics>& risk_metrics,
                           const std::vector<MarketData>& new_data) noexcept;

    // Risk limit management
    bool set_risk_limits(const std::unordered_map<std::string, float>& limits);
    std::unordered_map<std::string, float> get_risk_limits() const noexcept;

    // Stress testing
    std::vector<RiskMetrics> run_stress_tests(const std::vector<Order>& positions,
                                            const std::vector<MarketData>& baseline_data);

    // Risk reporting
    struct RiskReport {
        float total_var_95;
        float total_cvar_95;
        float portfolio_volatility;
        float max_drawdown;
        size_t positions_at_risk;
        std::vector<std::string> risk_warnings;
        Timestamp generated_at;
    };

    RiskReport generate_risk_report(const std::vector<RiskMetrics>& risk_metrics) const;

    // Performance monitoring
    PerformanceCounters get_performance_counters() const noexcept {
        return performance_counters_;
    }

    void reset_performance_counters() noexcept {
        performance_counters_ = PerformanceCounters{};
    }

private:
    // Risk calculation engines
    std::unique_ptr<VaRCalculator> var_calculator_;
    std::unique_ptr<PositionManager> position_manager_;
    std::unique_ptr<StressTester> stress_tester_;

    // Risk limits configuration
    struct RiskLimits {
        float max_portfolio_var = 0.05f;      // 5% VaR limit
        float max_position_var = 0.02f;       // 2% per position VaR
        float max_drawdown = 0.10f;           // 10% max drawdown
        float max_leverage = 5.0f;            // 5x max leverage
        float max_concentration = 0.20f;      // 20% max position size
        size_t max_positions = 100;           // Max number of positions
    } risk_limits_;

    // Performance tracking
    PerformanceCounters performance_counters_;

    // Risk calculation buffers
    std::vector<float> return_buffer_;
    std::vector<float> var_buffer_;
    std::unordered_map<SymbolId, RiskMetrics> risk_cache_;

    // Fast risk calculation helpers
    float fast_var_calculation(const std::vector<float>& returns,
                             float confidence_level = 0.95f) noexcept;
    float fast_cvar_calculation(const std::vector<float>& returns,
                              float confidence_level = 0.95f) noexcept;
    float fast_sharpe_ratio(const std::vector<float>& returns,
                          float risk_free_rate = 0.02f) noexcept;

    // Position aggregation for portfolio risk
    std::vector<float> aggregate_position_returns(const std::vector<Order>& positions,
                                                const std::vector<MarketData>& market_data) noexcept;

    // Risk limit checking
    bool check_var_limit(float portfolio_var) const noexcept;
    bool check_drawdown_limit(float current_drawdown) const noexcept;
    bool check_leverage_limit(float current_leverage) const noexcept;
    bool check_concentration_limit(const std::vector<Order>& positions) const noexcept;
};

// VaR Calculator - Ultra-fast Value at Risk calculations
class VaRCalculator {
public:
    VaRCalculator();
    ~VaRCalculator() = default;

    // Historical VaR calculation
    float calculate_historical_var(const std::vector<float>& returns,
                                 float confidence_level = 0.95f) noexcept;

    // Parametric VaR (Normal distribution assumption)
    float calculate_parametric_var(float mean_return, float volatility,
                                 float confidence_level = 0.95f) noexcept;

    // Monte Carlo VaR
    float calculate_monte_carlo_var(const std::vector<float>& returns,
                                  size_t n_simulations = 10000,
                                  float confidence_level = 0.95f) noexcept;

    // CVaR (Conditional VaR / Expected Shortfall)
    float calculate_cvar(const std::vector<float>& returns,
                        float confidence_level = 0.95f) noexcept;

    // Fast approximations for real-time use
    float fast_var_approximation(const std::vector<float>& returns,
                               float confidence_level = 0.95f) noexcept;

private:
    // Pre-computed normal distribution quantiles
    static constexpr std::array<float, 101> normal_quantiles_ = {
        -3.7190f, -3.0902f, -2.8070f, -2.6437f, -2.5244f, -2.4324f, -2.3570f, -2.2936f,
        -2.2389f, -2.1910f, -2.1483f, -2.1095f, -2.0742f, -2.0417f, -2.0117f, -1.9840f,
        -1.9580f, -1.9339f, -1.9111f, -1.8897f, -1.8697f, -1.8507f, -1.8326f, -1.8153f,
        -1.7988f, -1.7829f, -1.7677f, -1.7532f, -1.7393f, -1.7259f, -1.7131f, -1.7009f,
        -1.6892f, -1.6780f, -1.6673f, -1.6571f, -1.6473f, -1.6380f, -1.6291f, -1.6206f,
        -1.6125f, -1.6048f, -1.5974f, -1.5904f, -1.5837f, -1.5773f, -1.5712f, -1.5654f,
        -1.5599f, -1.5547f, -1.5497f, -1.5450f, -1.5406f, -1.5364f, -1.5324f, -1.5286f,
        -1.5250f, -1.5216f, -1.5184f, -1.5154f, -1.5126f, -1.5099f, -1.5073f, -1.5050f,
        -1.5027f, -1.5006f, -1.4986f, -1.4967f, -1.4950f, -1.4933f, -1.4918f, -1.4903f,
        -1.4890f, -1.4877f, -1.4866f, -1.4855f, -1.4845f, -1.4836f, -1.4827f, -1.4820f,
        -1.4812f, -1.4806f, -1.4800f, -1.4795f, -1.4790f, -1.4786f, -1.4782f, -1.4779f,
        -1.4776f, -1.4773f, -1.4771f, -1.4769f, -1.4767f, -1.4765f, -1.4764f, -1.4763f,
        -1.4762f, -1.4761f, -1.4760f, -1.4760f, -1.4759f, -1.4759f, -1.4759f, -1.4759f,
        0.0000f
    };

    // Random number generation for Monte Carlo
    std::mt19937 rng_;
    std::normal_distribution<float> normal_dist_;
};

// Position Manager - Real-time position tracking and management
class PositionManager {
public:
    PositionManager();
    ~PositionManager() = default;

    // Position management
    void add_position(const Order& order);
    void update_position(const Order& fill_update);
    void remove_position(OrderId order_id);
    void clear_all_positions();

    // Position queries
    std::vector<Order> get_all_positions() const noexcept;
    std::vector<Order> get_positions_by_symbol(SymbolId symbol_id) const noexcept;
    Order get_position(OrderId order_id) const noexcept;

    // Position aggregation
    struct PortfolioSummary {
        size_t total_positions;
        size_t long_positions;
        size_t short_positions;
        Price total_exposure;
        Price total_pnl;
        float gross_exposure;
        float net_exposure;
        std::unordered_map<SymbolId, size_t> positions_by_symbol;
    };

    PortfolioSummary get_portfolio_summary() const noexcept;

    // Risk-based position limits
    bool check_position_limits(const Order& new_order,
                            const PortfolioSummary& current_portfolio) const noexcept;

private:
    // Position storage with fast lookup
    std::unordered_map<OrderId, Order> positions_;
    std::unordered_map<SymbolId, std::vector<OrderId>> positions_by_symbol_;

    // Position limits
    struct PositionLimits {
        size_t max_positions_per_symbol = 5;
        Price max_exposure_per_symbol = price_from_double(1000000);  // $1M
        float max_portfolio_concentration = 0.25f;  // 25%
    } limits_;

    // Fast position aggregation
    void update_portfolio_cache() noexcept;
    PortfolioSummary cached_summary_;
    bool cache_valid_;
};

// Stress Tester - Portfolio stress testing under various scenarios
class StressTester {
public:
    StressTester();
    ~StressTester() = default;

    // Stress test scenarios
    enum class StressScenario {
        MARKET_CRASH,      // -20% across all assets
        SECTOR_CRASH,      // -30% in specific sector
        VOLATILITY_SPIKE,  // 3x volatility increase
        LIQUIDITY_CRISIS,  // Wide spreads, low volume
        FLASH_CRASH,       // Sudden price drops and recovery
        INTEREST_RATE_SHOCK, // Rate changes affecting valuations
        CURRENCY_CRISIS,   // FX volatility spike
        CUSTOM_SCENARIO    // User-defined scenario
    };

    // Run stress tests
    std::vector<RiskMetrics> run_stress_test(const std::vector<Order>& positions,
                                           const std::vector<MarketData>& baseline_data,
                                           StressScenario scenario);

    std::vector<RiskMetrics> run_multiple_stress_tests(const std::vector<Order>& positions,
                                                     const std::vector<MarketData>& baseline_data,
                                                     const std::vector<StressScenario>& scenarios);

    // Custom scenario definition
    struct CustomScenario {
        std::string name;
        std::unordered_map<SymbolId, float> price_shocks;  // Percentage changes
        std::unordered_map<SymbolId, float> volatility_shocks;  // Multipliers
        float market_correlation_change;  // Correlation adjustment
        Timestamp duration_ns;  // Scenario duration
    };

    std::vector<RiskMetrics> run_custom_stress_test(const std::vector<Order>& positions,
                                                  const std::vector<MarketData>& baseline_data,
                                                  const CustomScenario& scenario);

private:
    // Scenario generation
    std::vector<MarketData> generate_stress_market_data(const std::vector<MarketData>& baseline,
                                                      StressScenario scenario) noexcept;

    std::vector<MarketData> apply_price_shocks(const std::vector<MarketData>& data,
                                             const std::unordered_map<SymbolId, float>& shocks) noexcept;

    std::vector<MarketData> apply_volatility_shocks(const std::vector<MarketData>& data,
                                                  const std::unordered_map<SymbolId, float>& shocks) noexcept;

    // Scenario-specific implementations
    std::vector<MarketData> generate_market_crash_scenario(const std::vector<MarketData>& baseline) noexcept;
    std::vector<MarketData> generate_flash_crash_scenario(const std::vector<MarketData>& baseline) noexcept;
    std::vector<MarketData> generate_volatility_spike_scenario(const std::vector<MarketData>& baseline) noexcept;
};

// Utility functions for risk calculations
namespace risk_utils {

// Fast correlation calculation
float correlation_coefficient(const std::vector<float>& x, const std::vector<float>& y) noexcept;

// Fast covariance calculation
float covariance(const std::vector<float>& x, const std::vector<float>& y) noexcept;

// Portfolio volatility calculation
float portfolio_volatility(const std::vector<float>& weights,
                          const std::vector<std::vector<float>>& covariance_matrix) noexcept;

// Beta calculation relative to market
float calculate_beta(const std::vector<float>& asset_returns,
                    const std::vector<float>& market_returns) noexcept;

// Value at Risk utility functions
float percentile(const std::vector<float>& data, float percentile) noexcept;
std::vector<float> sort_returns(const std::vector<float>& returns) noexcept;

// Risk-adjusted return metrics
float sortino_ratio(const std::vector<float>& returns, float target_return = 0.0f) noexcept;
float calmar_ratio(const std::vector<float>& returns, float max_drawdown) noexcept;
float information_ratio(const std::vector<float>& returns, const std::vector<float>& benchmark) noexcept;

} // namespace risk_utils

} // namespace trading
