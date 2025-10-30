#pragma once

#include "../core/types.hpp"
#include "../core/quantization.hpp"
#include <vector>
#include <array>
#include <memory>
#include <functional>

namespace trading {

// Forward declarations
class TechnicalIndicators;
class StatisticalSignals;
class MLSignals;

class SignalEngine {
public:
    SignalEngine();
    ~SignalEngine() = default;

    // Disable copy/move for performance
    SignalEngine(const SignalEngine&) = delete;
    SignalEngine& operator=(const SignalEngine&) = delete;
    SignalEngine(SignalEngine&&) = delete;
    SignalEngine& operator=(SignalEngine&&) = delete;

    // Core signal generation
    std::vector<Signal> generate_signals(const std::vector<MarketData>& market_data);
    std::vector<Signal> generate_signals_simd(const std::vector<MarketData>& market_data);

    // Strategy-specific signal generation
    std::vector<Signal> generate_momentum_signals(const std::vector<MarketData>& market_data);
    std::vector<Signal> generate_mean_reversion_signals(const std::vector<MarketData>& market_data);
    std::vector<Signal> generate_volatility_signals(const std::vector<MarketData>& market_data);
    std::vector<Signal> generate_arbitrage_signals(const std::vector<MarketData>& market_data);

    // Advanced signal processing
    std::vector<Signal> combine_signals(const std::vector<std::vector<Signal>>& signal_vectors);
    std::vector<Signal> filter_signals(const std::vector<Signal>& signals,
                                     const std::function<bool(const Signal&)>& filter_func);

    // Signal strength and confidence calculation
    void update_signal_confidence(std::vector<Signal>& signals);
    void normalize_signal_strength(std::vector<Signal>& signals);

    // Performance monitoring
    PerformanceCounters get_performance_counters() const noexcept {
        return performance_counters_;
    }

    void reset_performance_counters() noexcept {
        performance_counters_ = PerformanceCounters{};
    }

private:
    // Sub-engines for different signal types
    std::unique_ptr<TechnicalIndicators> technical_indicators_;
    std::unique_ptr<StatisticalSignals> statistical_signals_;
    std::unique_ptr<MLSignals> ml_signals_;

    // Performance tracking
    PerformanceCounters performance_counters_;

    // Internal buffers for SIMD processing
    std::vector<MarketData::Quantized> quantized_data_buffer_;
    std::vector<Signal::Quantized> quantized_signal_buffer_;

    // Pre-computed lookup tables for fast signal generation
    std::array<std::array<float, 256>, MAX_SYMBOLS> signal_history_cache_;

    // Configuration parameters
    struct Config {
        size_t max_signals_per_symbol = 10;
        float min_signal_confidence = 0.1f;
        float max_signal_age_ns = 1000000000;  // 1 second
        bool use_simd = true;
        bool use_quantization = true;
        bool enable_signal_filtering = true;
    } config_;

    // Internal helper methods
    void preprocess_market_data(const std::vector<MarketData>& market_data);
    void generate_technical_signals(std::vector<Signal>& signals);
    void generate_statistical_signals(std::vector<Signal>& signals);
    void generate_ml_signals(std::vector<Signal>& signals);

    // SIMD-optimized processing
    void process_simd_batch(const MarketData::Quantized* data_batch,
                           Signal::Quantized* signal_batch,
                           size_t batch_size);

    // Signal validation and filtering
    bool validate_signal(const Signal& signal) const noexcept;
    void apply_signal_filters(std::vector<Signal>& signals);

    // Memory management for high-frequency operation
    void prefetch_data_for_symbol(SymbolId symbol_id) noexcept;
    void update_signal_cache(SymbolId symbol_id, const Signal& signal) noexcept;
};

// Technical indicators engine
class TechnicalIndicators {
public:
    TechnicalIndicators() = default;
    ~TechnicalIndicators() = default;

    // Core technical indicators (SIMD optimized where possible)
    struct IndicatorValues {
        float sma_20;      // Simple moving average 20
        float sma_50;      // Simple moving average 50
        float ema_12;      // Exponential moving average 12
        float ema_26;      // Exponential moving average 26
        float rsi;         // Relative strength index
        float macd;        // MACD line
        float macd_signal; // MACD signal line
        float macd_hist;   // MACD histogram
        float bb_upper;    // Bollinger band upper
        float bb_lower;    // Bollinger band lower
        float bb_middle;   // Bollinger band middle
        float atr;         // Average true range
        float stoch_k;     // Stochastic K
        float stoch_d;     // Stochastic D
    };

    IndicatorValues calculate_indicators(const std::vector<Price>& prices,
                                       const std::vector<Volume>& volumes) noexcept;

    // Fast approximation versions for real-time processing
    float fast_rsi(const std::vector<Price>& prices) noexcept;
    float fast_macd(const std::vector<Price>& prices) noexcept;
    std::pair<float, float> fast_bollinger_bands(const std::vector<Price>& prices) noexcept;

private:
    // Pre-allocated buffers for indicator calculations
    std::vector<float> price_buffer_;
    std::vector<float> volume_buffer_;
    std::vector<float> gain_buffer_;
    std::vector<float> loss_buffer_;

    // Lookup tables for common calculations
    std::array<float, 256> rsi_lookup_;
    std::array<float, 256> ema_weights_;

    void initialize_lookup_tables();
};

// Statistical signals engine
class StatisticalSignals {
public:
    StatisticalSignals() = default;
    ~StatisticalSignals() = default;

    // Statistical signal generation
    struct StatisticalFeatures {
        float mean_return;
        float return_volatility;
        float skewness;
        float kurtosis;
        float autocorrelation_1;
        float autocorrelation_5;
        float hurst_exponent;
        float fractal_dimension;
        float entropy;
        float mutual_information;
    };

    StatisticalFeatures calculate_features(const std::vector<Price>& prices) noexcept;

    // Signal generation from statistical features
    Signal generate_momentum_signal(SymbolId symbol_id, const StatisticalFeatures& features) noexcept;
    Signal generate_mean_reversion_signal(SymbolId symbol_id, const StatisticalFeatures& features) noexcept;
    Signal generate_volatility_signal(SymbolId symbol_id, const StatisticalFeatures& features) noexcept;

    // Fast statistical approximations
    float fast_autocorrelation(const std::vector<Price>& prices, int lag) noexcept;
    float fast_hurst_exponent(const std::vector<Price>& prices) noexcept;
    float fast_entropy(const std::vector<Price>& prices) noexcept;

private:
    // Statistical computation buffers
    std::vector<float> return_buffer_;
    std::vector<float> feature_buffer_;

    // Pre-computed statistical constants
    static constexpr float NORMAL_SKEWNESS_THRESHOLD = 0.5f;
    static constexpr float NORMAL_KURTOSIS_THRESHOLD = 0.5f;
    static constexpr float HIGH_VOLATILITY_THRESHOLD = 0.03f;
};

// ML signals engine (for advanced strategies)
class MLSignals {
public:
    MLSignals();
    ~MLSignals() = default;

    // ML-based signal generation
    Signal generate_attention_signal(SymbolId symbol_id,
                                   const std::vector<MarketData>& recent_data) noexcept;
    Signal generate_sentiment_signal(SymbolId symbol_id,
                                   const std::vector<MarketData>& recent_data) noexcept;
    Signal generate_complex_systems_signal(SymbolId symbol_id,
                                         const std::vector<MarketData>& recent_data) noexcept;

    // Neural network approximations (simplified for speed)
    float fast_neural_network(const std::vector<float>& inputs,
                            const std::vector<std::vector<float>>& weights) noexcept;
    float fast_sigmoid(float x) noexcept;
    float fast_tanh(float x) noexcept;

private:
    // Simplified neural network weights (pre-trained)
    std::vector<std::vector<float>> attention_weights_;
    std::vector<std::vector<float>> sentiment_weights_;
    std::vector<std::vector<float>> complex_weights_;

    void initialize_neural_weights();
};

// Utility functions for signal processing
namespace signal_utils {

// Signal aggregation functions
Signal aggregate_signals(const std::vector<Signal>& signals) noexcept;

// Signal confidence calculation
float calculate_signal_confidence(const Signal& signal,
                                const std::vector<MarketData>& historical_data) noexcept;

// Signal decay function (older signals have lower weight)
float signal_decay_factor(Timestamp signal_time, Timestamp current_time) noexcept;

// Cross-signal correlation analysis
float signal_correlation(const std::vector<Signal>& signals_a,
                        const std::vector<Signal>& signals_b) noexcept;

// Signal quality metrics
struct SignalQuality {
    float consistency;    // How consistent is the signal over time
    float reliability;    // Historical success rate
    float timeliness;     // How timely the signal is
    float strength;       // Overall signal strength
};

SignalQuality assess_signal_quality(const Signal& signal,
                                  const std::vector<Signal>& historical_signals) noexcept;

} // namespace signal_utils

} // namespace trading
