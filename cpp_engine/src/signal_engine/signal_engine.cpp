#include "signal_engine.hpp"
#include "../core/memory_pool.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace trading {

SignalEngine::SignalEngine()
    : technical_indicators_(std::make_unique<TechnicalIndicators>()),
      statistical_signals_(std::make_unique<StatisticalSignals>()),
      ml_signals_(std::make_unique<MLSignals>()) {

    // Initialize signal history cache
    signal_history_cache_ = {};
    quantized_data_buffer_.reserve(1024);
    quantized_signal_buffer_.reserve(1024);

    // Pre-warm cache with neutral signals
    for (auto& symbol_cache : signal_history_cache_) {
        symbol_cache.fill(0.0f);
    }
}

std::vector<Signal> SignalEngine::generate_signals(const std::vector<MarketData>& market_data) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<Signal> signals;

    if (config_.use_simd && market_data.size() >= 8) {
        // Use SIMD-optimized processing for large batches
        signals = generate_signals_simd(market_data);
    } else {
        // Use standard processing
        preprocess_market_data(market_data);

        // Reserve space for signals
        signals.reserve(market_data.size() * config_.max_signals_per_symbol);

        // Generate different types of signals
        generate_technical_signals(signals);
        generate_statistical_signals(signals);
        generate_ml_signals(signals);

        // Post-process signals
        if (config_.enable_signal_filtering) {
            apply_signal_filters(signals);
        }

        update_signal_confidence(signals);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    performance_counters_.signals_processed += signals.size();
    performance_counters_.update_signal_latency(latency);

    return signals;
}

std::vector<Signal> SignalEngine::generate_signals_simd(const std::vector<MarketData>& market_data) {
    std::vector<Signal> signals;
    signals.reserve(market_data.size() * 2);  // Estimate 2 signals per data point

    // Quantize market data for SIMD processing
    quantized_data_buffer_.clear();
    quantized_data_buffer_.reserve(market_data.size());

    for (const auto& data : market_data) {
        quantized_data_buffer_.push_back(data.quantize());
    }

    // Process in SIMD batches
    constexpr size_t SIMD_BATCH_SIZE = 8;  // AVX-512 processes 8 floats at once
    quantized_signal_buffer_.resize(quantized_data_buffer_.size() * 2);

    for (size_t i = 0; i < quantized_data_buffer_.size(); i += SIMD_BATCH_SIZE) {
        size_t batch_size = std::min(SIMD_BATCH_SIZE, quantized_data_buffer_.size() - i);
        process_simd_batch(
            &quantized_data_buffer_[i],
            &quantized_signal_buffer_[i * 2],
            batch_size
        );
    }

    // Convert quantized signals back to full precision
    for (const auto& qsignal : quantized_signal_buffer_) {
        if (qsignal.signal != 0) {  // Only include non-neutral signals
            Signal signal{
                qsignal.symbol_id,
                static_cast<SignalType>(qsignal.signal),
                signal_utils::dequantize_confidence(qsignal.confidence),
                qsignal.timestamp,
                qsignal.strategy_id,
                dequantize_price(qsignal.target_price),
                dequantize_quantity(qsignal.target_quantity)
            };
            signals.push_back(signal);
        }
    }

    return signals;
}

void SignalEngine::preprocess_market_data(const std::vector<MarketData>& market_data) {
    // Update signal history cache for each symbol
    for (const auto& data : market_data) {
        prefetch_data_for_symbol(data.symbol_id);

        // Simple moving average update for cache
        auto& cache = signal_history_cache_[data.symbol_id];
        float current_return = price_to_double(data.ask_price - data.bid_price) /
                              price_to_double((data.ask_price + data.bid_price) / 2);

        // Shift cache and add new return
        std::shift_right(cache.begin(), cache.end(), 1);
        cache[0] = current_return;
    }
}

void SignalEngine::generate_technical_signals(std::vector<Signal>& signals) {
    // Generate technical indicator based signals
    for (size_t i = 0; i < quantized_data_buffer_.size(); ++i) {
        const auto& data = quantized_data_buffer_[i];

        // Simple technical signals (can be expanded)
        QPrice spread = data.ask_price - data.bid_price;
        QPrice mid_price = (data.ask_price + data.bid_price) / 2;

        // Tight spread signal (liquidity)
        if (spread < quantize_price(price_from_double(0.01))) {  // Spread < $0.01
            signals.push_back({
                data.symbol_id,
                SignalType::LONG,
                0.7f,
                data.timestamp,
                1,  // Technical strategy ID
                mid_price,
                100  // Default quantity
            });
        }

        // Wide spread signal (caution)
        if (spread > quantize_price(price_from_double(0.10))) {  // Spread > $0.10
            signals.push_back({
                data.symbol_id,
                SignalType::NEUTRAL,
                0.8f,
                data.timestamp,
                1,
                mid_price,
                0
            });
        }
    }
}

void SignalEngine::generate_statistical_signals(std::vector<Signal>& signals) {
    // Generate statistical arbitrage signals
    for (size_t i = 0; i < quantized_data_buffer_.size(); ++i) {
        const auto& data = quantized_data_buffer_[i];

        // Simple statistical signal based on price movement
        const auto& cache = signal_history_cache_[data.symbol_id];

        // Calculate simple momentum
        float recent_momentum = std::accumulate(cache.begin(), cache.begin() + 5, 0.0f) / 5.0f;
        float longer_momentum = std::accumulate(cache.begin(), cache.begin() + 20, 0.0f) / 20.0f;

        if (recent_momentum > longer_momentum * 1.1f) {  // Strong upward momentum
            signals.push_back({
                data.symbol_id,
                SignalType::LONG,
                0.6f,
                data.timestamp,
                2,  // Statistical strategy ID
                data.ask_price,
                50
            });
        } else if (recent_momentum < longer_momentum * 0.9f) {  // Strong downward momentum
            signals.push_back({
                data.symbol_id,
                SignalType::SHORT,
                0.6f,
                data.timestamp,
                2,
                data.bid_price,
                50
            });
        }
    }
}

void SignalEngine::generate_ml_signals(std::vector<Signal>& signals) {
    // Generate ML-based signals (simplified for performance)
    for (size_t i = 0; i < quantized_data_buffer_.size(); ++i) {
        const auto& data = quantized_data_buffer_[i];

        // Simple pattern recognition (can be replaced with actual ML)
        QPrice range = data.ask_price - data.bid_price;
        QQuantity volume_indicator = data.ask_size + data.bid_size;

        // Pattern-based signal
        if (range < quantize_price(price_from_double(0.02)) &&
            volume_indicator > 1000) {  // Tight range with good volume
            signals.push_back({
                data.symbol_id,
                SignalType::LONG,
                0.5f,
                data.timestamp,
                3,  // ML strategy ID
                data.ask_price,
                25
            });
        }
    }
}

void SignalEngine::process_simd_batch(const MarketData::Quantized* data_batch,
                                    Signal::Quantized* signal_batch,
                                    size_t batch_size) {
#ifdef __AVX512F__
    // AVX-512 SIMD processing for maximum performance
    __m512i bid_prices = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data_batch->bid_price));
    __m512i ask_prices = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data_batch->ask_price));

    // Calculate spreads using SIMD
    __m512i spreads = _mm512_sub_epi32(ask_prices, bid_prices);

    // Threshold comparison for signal generation
    __m512i tight_spread_threshold = _mm512_set1_epi32(quantize_price(price_from_double(0.01)));
    __mmask8 tight_spread_mask = _mm512_cmplt_epi32_mask(spreads, tight_spread_threshold);

    // Generate signals for tight spreads
    if (tight_spread_mask) {
        for (size_t j = 0; j < batch_size; ++j) {
            if (tight_spread_mask & (1 << j)) {
                signal_batch[j * 2] = {
                    data_batch[j].symbol_id,
                    1,  // LONG
                    180,  // ~0.7 confidence
                    data_batch[j].timestamp,
                    1,  // Technical strategy
                    (data_batch[j].bid_price + data_batch[j].ask_price) / 2,
                    100
                };
            }
        }
    }
#else
    // Fallback for systems without AVX-512
    for (size_t j = 0; j < batch_size; ++j) {
        QPrice spread = data_batch[j].ask_price - data_batch[j].bid_price;
        QPrice tight_threshold = quantize_price(price_from_double(0.01));

        if (spread < tight_threshold) {
            signal_batch[j * 2] = {
                data_batch[j].symbol_id,
                1,  // LONG
                180,  // ~0.7 confidence
                data_batch[j].timestamp,
                1,  // Technical strategy
                (data_batch[j].bid_price + data_batch[j].ask_price) / 2,
                100
            };
        }
    }
#endif
}

void SignalEngine::apply_signal_filters(std::vector<Signal>& signals) {
    // Remove old signals
    Timestamp cutoff_time = current_timestamp() - static_cast<Timestamp>(config_.max_signal_age_ns);

    auto it = std::remove_if(signals.begin(), signals.end(),
        [cutoff_time](const Signal& s) { return s.timestamp < cutoff_time; });
    signals.erase(it, signals.end());

    // Remove low-confidence signals
    it = std::remove_if(signals.begin(), signals.end(),
        [this](const Signal& s) { return s.confidence < config_.min_signal_confidence; });
    signals.erase(it, signals.end());

    // Sort by confidence (highest first)
    std::sort(signals.begin(), signals.end(),
        [](const Signal& a, const Signal& b) { return a.confidence > b.confidence; });

    // Limit signals per symbol
    std::unordered_map<SymbolId, size_t> symbol_counts;
    it = std::remove_if(signals.begin(), signals.end(),
        [this, &symbol_counts](const Signal& s) {
            size_t& count = symbol_counts[s.symbol_id];
            if (count >= config_.max_signals_per_symbol) {
                return true;
            }
            count++;
            return false;
        });
    signals.erase(it, signals.end());
}

void SignalEngine::update_signal_confidence(std::vector<Signal>& signals) {
    // Update confidence based on signal history and market conditions
    for (auto& signal : signals) {
        // Base confidence adjustment
        float base_confidence = signal.confidence;

        // Historical performance adjustment (simplified)
        const auto& cache = signal_history_cache_[signal.symbol_id];
        float recent_volatility = 0.0f;
        for (size_t i = 0; i < 10 && i < cache.size(); ++i) {
            recent_volatility += std::abs(cache[i]);
        }
        recent_volatility /= 10.0f;

        // Reduce confidence in high volatility
        if (recent_volatility > 0.02f) {
            base_confidence *= 0.8f;
        }

        // Signal type specific adjustments
        if (signal.signal == SignalType::LONG && signal.target_price > signal.target_price * 1.05f) {
            base_confidence *= 0.9f;  // Reduce confidence for aggressive targets
        }

        signal.confidence = std::clamp(base_confidence, 0.0f, 1.0f);
    }
}

void SignalEngine::prefetch_data_for_symbol(SymbolId symbol_id) noexcept {
    // Prefetch signal history cache for better memory performance
#ifdef __GNUC__
    __builtin_prefetch(&signal_history_cache_[symbol_id], 0, 3);
#endif
}

void SignalEngine::update_signal_cache(SymbolId symbol_id, const Signal& signal) noexcept {
    // Update cache with signal strength (simplified)
    float signal_strength = static_cast<float>(signal.signal) * signal.confidence;
    auto& cache = signal_history_cache_[symbol_id];

    // Shift and add new signal
    std::shift_right(cache.begin(), cache.end(), 1);
    cache[0] = signal_strength;
}

// Technical Indicators Implementation
void TechnicalIndicators::initialize_lookup_tables() {
    // Initialize RSI lookup table
    for (size_t i = 0; i < rsi_lookup_.size(); ++i) {
        float gain_ratio = static_cast<float>(i) / 255.0f;
        rsi_lookup_[i] = 100.0f - (100.0f / (1.0f + gain_ratio));
    }

    // Initialize EMA weights
    for (size_t i = 0; i < ema_weights_.size(); ++i) {
        float period = static_cast<float>(i + 1);
        ema_weights_[i] = 2.0f / (period + 1.0f);
    }
}

TechnicalIndicators::IndicatorValues TechnicalIndicators::calculate_indicators(
    const std::vector<Price>& prices, const std::vector<Volume>& volumes) noexcept {

    IndicatorValues result = {};

    if (prices.size() < 50) return result;  // Need minimum data

    size_t n = prices.size();

    // Convert to floats for calculation
    std::vector<float> price_floats(n);
    for (size_t i = 0; i < n; ++i) {
        price_floats[i] = price_to_double(prices[i]);
    }

    // Simple Moving Averages
    result.sma_20 = std::accumulate(price_floats.end() - 20, price_floats.end(), 0.0f) / 20.0f;
    result.sma_50 = std::accumulate(price_floats.end() - 50, price_floats.end(), 0.0f) / 50.0f;

    // Exponential Moving Averages (simplified)
    result.ema_12 = price_floats.back() * 0.1538f + result.sma_20 * 0.8462f;  // Approximation
    result.ema_26 = price_floats.back() * 0.0741f + result.sma_50 * 0.9259f;  // Approximation

    // RSI calculation (simplified)
    result.rsi = fast_rsi(price_floats);

    // MACD (simplified)
    result.macd = result.ema_12 - result.ema_26;
    result.macd_signal = result.macd * 0.8f;  // Approximation
    result.macd_hist = result.macd - result.macd_signal;

    // Bollinger Bands
    auto [upper, lower] = fast_bollinger_bands(price_floats);
    result.bb_upper = upper;
    result.bb_lower = lower;
    result.bb_middle = result.sma_20;

    return result;
}

float TechnicalIndicators::fast_rsi(const std::vector<float>& prices) noexcept {
    if (prices.size() < 14) return 50.0f;

    float gains = 0.0f, losses = 0.0f;

    for (size_t i = 1; i < 14; ++i) {
        float change = prices[i] - prices[i-1];
        if (change > 0) gains += change;
        else losses -= change;
    }

    if (losses == 0.0f) return 100.0f;
    float rs = gains / losses;

    return 100.0f - (100.0f / (1.0f + rs));
}

std::pair<float, float> TechnicalIndicators::fast_bollinger_bands(const std::vector<float>& prices) noexcept {
    if (prices.size() < 20) return {0.0f, 0.0f};

    // Calculate SMA
    float sma = std::accumulate(prices.end() - 20, prices.end(), 0.0f) / 20.0f;

    // Calculate standard deviation
    float variance = 0.0f;
    for (size_t i = prices.size() - 20; i < prices.size(); ++i) {
        float diff = prices[i] - sma;
        variance += diff * diff;
    }
    variance /= 19.0f;  // Sample variance
    float std_dev = std::sqrt(variance);

    return {sma + 2 * std_dev, sma - 2 * std_dev};
}

} // namespace trading
