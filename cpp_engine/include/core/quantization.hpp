#pragma once

#include "types.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace trading {

// Quantization utilities for performance optimization
// Trade precision for speed in high-frequency computations

class PriceQuantizer {
private:
    static constexpr double SCALE_FACTOR = 10000.0;  // 4 decimal places
    static constexpr int32_t MAX_QUANTIZED = std::numeric_limits<int32_t>::max() / 2;
    static constexpr int32_t MIN_QUANTIZED = std::numeric_limits<int32_t>::min() / 2;

public:
    // Quantize price to int32 for fast computation
    static QPrice quantize(Price price) noexcept {
        double scaled = static_cast<double>(price) / PRICE_SCALE;
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        return std::clamp(quantized, MIN_QUANTIZED, MAX_QUANTIZED);
    }

    // Dequantize back to full precision
    static Price dequantize(QPrice qprice) noexcept {
        return static_cast<Price>(qprice) * PRICE_SCALE;
    }

    // Fast arithmetic operations on quantized prices
    static QPrice add(QPrice a, QPrice b) noexcept {
        return a + b;
    }

    static QPrice subtract(QPrice a, QPrice b) noexcept {
        return a - b;
    }

    static QPrice multiply(QPrice a, QPrice b) noexcept {
        // Use 64-bit intermediate to avoid overflow
        int64_t result = static_cast<int64_t>(a) * static_cast<int64_t>(b);
        return static_cast<QPrice>(result / SCALE_FACTOR);
    }

    static QPrice divide(QPrice a, QPrice b) noexcept {
        if (b == 0) return 0;
        int64_t result = (static_cast<int64_t>(a) * SCALE_FACTOR) / static_cast<int64_t>(b);
        return static_cast<QPrice>(result);
    }

    // Comparison operations
    static bool greater_than(QPrice a, QPrice b) noexcept {
        return a > b;
    }

    static bool less_than(QPrice a, QPrice b) noexcept {
        return a < b;
    }

    static bool equal(QPrice a, QPrice b) noexcept {
        return a == b;
    }

    // Distance calculation (Manhattan)
    static QPrice manhattan_distance(QPrice a, QPrice b) noexcept {
        return std::abs(a - b);
    }

    // Percentage change calculation
    static QPrice percentage_change(QPrice old_price, QPrice new_price) noexcept {
        if (old_price == 0) return 0;
        int64_t change = (static_cast<int64_t>(new_price - old_price) * SCALE_FACTOR * 100) /
                        static_cast<int64_t>(old_price);
        return static_cast<QPrice>(change);
    }
};

class SignalQuantizer {
private:
    static constexpr float CONFIDENCE_SCALE = 255.0f;  // 8-bit quantization

public:
    // Quantize signal confidence to uint8
    static uint8_t quantize_confidence(float confidence) noexcept {
        return static_cast<uint8_t>(std::clamp(confidence * CONFIDENCE_SCALE, 0.0f, 255.0f));
    }

    // Dequantize confidence back to float
    static float dequantize_confidence(uint8_t qconfidence) noexcept {
        return static_cast<float>(qconfidence) / CONFIDENCE_SCALE;
    }

    // Fast signal strength calculation
    static uint8_t signal_strength(uint8_t confidence, int8_t direction) noexcept {
        // Combine confidence and direction into a single byte
        uint8_t abs_direction = static_cast<uint8_t>(std::abs(direction));
        return (confidence * abs_direction) / 3;  // Scale down to prevent overflow
    }
};

class VolumeQuantizer {
private:
    static constexpr int32_t MAX_VOLUME = 10000000;  // 10M shares max
    static constexpr int16_t MAX_QUANTIZED = std::numeric_limits<int16_t>::max();

public:
    // Quantize volume to int16
    static QQuantity quantize(Quantity volume) noexcept {
        if (volume <= 0) return 0;
        if (volume >= MAX_VOLUME) return MAX_QUANTIZED;

        // Use logarithmic quantization for large volumes
        if (volume > 10000) {
            float log_vol = std::log(static_cast<float>(volume));
            return static_cast<QQuantity>(log_vol * 1000.0f);
        } else {
            return static_cast<QQuantity>(volume);
        }
    }

    // Dequantize volume
    static Quantity dequantize(QQuantity qvolume) noexcept {
        if (qvolume <= 0) return 0;
        if (qvolume >= MAX_QUANTIZED) return MAX_VOLUME;

        // Reverse logarithmic quantization
        if (qvolume > 10000) {
            float exp_vol = std::exp(static_cast<float>(qvolume) / 1000.0f);
            return static_cast<Quantity>(exp_vol);
        } else {
            return static_cast<Quantity>(qvolume);
        }
    }

    // Fast volume comparison
    static bool sufficient_volume(QQuantity available, QQuantity required) noexcept {
        return available >= required;
    }

    // Volume-weighted average price calculation
    static QPrice volume_weighted_price(QPrice price1, QQuantity vol1,
                                      QPrice price2, QQuantity vol2) noexcept {
        int64_t total_volume = static_cast<int64_t>(vol1) + static_cast<int64_t>(vol2);
        if (total_volume == 0) return 0;

        int64_t weighted_sum = static_cast<int64_t>(price1) * static_cast<int64_t>(vol1) +
                             static_cast<int64_t>(price2) * static_cast<int64_t>(vol2);

        return static_cast<QPrice>(weighted_sum / total_volume);
    }
};

// Fast approximation functions for expensive operations
class FastApproximations {
public:
    // Fast exponential approximation (relative error < 1%)
    static float fast_exp(float x) noexcept {
        // Approximation: exp(x) ≈ (1 + x/1024)^1024
        // For |x| < 1, use Taylor series approximation
        if (std::abs(x) < 1.0f) {
            return 1.0f + x + x*x*0.5f + x*x*x*0.1666667f;
        } else if (x > 0) {
            return 2.7182818f;  // e^1 approximation
        } else {
            return 0.3678794f;  // e^-1 approximation
        }
    }

    // Fast logarithm approximation (relative error < 2%)
    static float fast_log(float x) noexcept {
        if (x <= 0) return -std::numeric_limits<float>::infinity();

        // Use approximation: log(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3
        float y = x - 1.0f;
        return y - y*y*0.5f + y*y*y*0.3333333f;
    }

    // Fast square root using Newton's method approximation
    static float fast_sqrt(float x) noexcept {
        if (x <= 0) return 0;

        float guess = x * 0.5f;  // Initial guess
        guess = (guess + x / guess) * 0.5f;  // One Newton iteration
        return guess;
    }

    // Fast normal CDF approximation (for risk calculations)
    static float fast_normal_cdf(float x) noexcept {
        // Abramowitz & Stegun approximation
        static constexpr float a1 =  0.886226899f;
        static constexpr float a2 = -1.645349621f;
        static constexpr float a3 =  0.914624893f;
        static constexpr float a4 = -0.140543331f;

        float y = 1.0f / (1.0f + 0.2316419f * std::abs(x));
        float z = fast_exp(-x * x * 0.5f);

        float cdf = 1.0f - z * (a1 * y + a2 * y*y + a3 * y*y*y + a4 * y*y*y*y);

        return x > 0 ? cdf : 1.0f - cdf;
    }

    // Fast variance calculation using Welford's online algorithm
    class FastVariance {
    private:
        int64_t count = 0;
        double mean = 0.0;
        double m2 = 0.0;

    public:
        void update(double value) noexcept {
            count++;
            double delta = value - mean;
            mean += delta / count;
            double delta2 = value - mean;
            m2 += delta * delta2;
        }

        double variance() const noexcept {
            return count > 1 ? m2 / (count - 1) : 0.0;
        }

        double std_dev() const noexcept {
            return fast_sqrt(variance());
        }

        void reset() noexcept {
            count = 0;
            mean = 0.0;
            m2 = 0.0;
        }
    };
};

// Lookup tables for expensive computations
class LookupTables {
private:
    static constexpr size_t EXP_TABLE_SIZE = 1024;
    static constexpr size_t LOG_TABLE_SIZE = 1024;
    static constexpr size_t SQRT_TABLE_SIZE = 1024;

    std::array<float, EXP_TABLE_SIZE> exp_table;
    std::array<float, LOG_TABLE_SIZE> log_table;
    std::array<float, SQRT_TABLE_SIZE> sqrt_table;

    static constexpr float EXP_MIN = -5.0f;
    static constexpr float EXP_MAX = 5.0f;
    static constexpr float LOG_MIN = 0.001f;
    static constexpr float LOG_MAX = 1000.0f;
    static constexpr float SQRT_MAX = 1000.0f;

public:
    LookupTables() {
        // Initialize lookup tables
        for (size_t i = 0; i < EXP_TABLE_SIZE; ++i) {
            float x = EXP_MIN + (EXP_MAX - EXP_MIN) * i / (EXP_TABLE_SIZE - 1);
            exp_table[i] = FastApproximations::fast_exp(x);
        }

        for (size_t i = 0; i < LOG_TABLE_SIZE; ++i) {
            float x = LOG_MIN + (LOG_MAX - LOG_MIN) * i / (LOG_TABLE_SIZE - 1);
            log_table[i] = FastApproximations::fast_log(x);
        }

        for (size_t i = 0; i < SQRT_TABLE_SIZE; ++i) {
            float x = SQRT_MAX * i / (SQRT_TABLE_SIZE - 1);
            sqrt_table[i] = FastApproximations::fast_sqrt(x);
        }
    }

    float lookup_exp(float x) const noexcept {
        if (x < EXP_MIN) return FastApproximations::fast_exp(EXP_MIN);
        if (x > EXP_MAX) return FastApproximations::fast_exp(EXP_MAX);

        size_t index = static_cast<size_t>((x - EXP_MIN) / (EXP_MAX - EXP_MIN) * (EXP_TABLE_SIZE - 1));
        index = std::min(index, EXP_TABLE_SIZE - 1);
        return exp_table[index];
    }

    float lookup_log(float x) const noexcept {
        if (x < LOG_MIN) return FastApproximations::fast_log(LOG_MIN);
        if (x > LOG_MAX) return FastApproximations::fast_log(LOG_MAX);

        size_t index = static_cast<size_t>((x - LOG_MIN) / (LOG_MAX - LOG_MIN) * (LOG_TABLE_SIZE - 1));
        index = std::min(index, LOG_TABLE_SIZE - 1);
        return log_table[index];
    }

    float lookup_sqrt(float x) const noexcept {
        if (x < 0) return 0;
        if (x > SQRT_MAX) return FastApproximations::fast_sqrt(x);

        size_t index = static_cast<size_t>(x / SQRT_MAX * (SQRT_TABLE_SIZE - 1));
        index = std::min(index, SQRT_TABLE_SIZE - 1);
        return sqrt_table[index];
    }
};

// Global lookup table instance
inline const LookupTables& get_lookup_tables() {
    static LookupTables tables;
    return tables;
}

} // namespace trading
