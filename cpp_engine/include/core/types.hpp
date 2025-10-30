#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>

namespace trading {

// Basic types with explicit sizes for cross-platform compatibility
using Price = int64_t;        // Price in fixed-point representation (e.g., * 10000 for 4 decimal places)
using Quantity = int32_t;     // Quantity as integer
using Timestamp = uint64_t;   // Nanoseconds since epoch
using OrderId = uint64_t;     // Unique order identifier
using SymbolId = uint32_t;    // Internal symbol identifier
using VenueId = uint16_t;     // Trading venue identifier

// Fixed-point arithmetic constants
constexpr int64_t PRICE_SCALE = 10000;  // 4 decimal places
constexpr int64_t PRICE_MULTIPLIER = PRICE_SCALE;
constexpr int64_t PRICE_DIVISOR = PRICE_SCALE;

// Quantized types for ultra-fast computation
using QPrice = int32_t;       // Quantized price (lossy compression)
using QQuantity = int16_t;    // Quantized quantity (lossy compression)

// SIMD-friendly data structures
struct alignas(64) MarketData {
    SymbolId symbol_id;
    Price bid_price;
    Price ask_price;
    Quantity bid_size;
    Quantity ask_size;
    Timestamp timestamp;
    uint32_t venue_id;
    uint16_t flags;  // Various market data flags

    // SIMD-friendly quantized version
    struct Quantized {
        SymbolId symbol_id;
        QPrice bid_price;
        QPrice ask_price;
        QQuantity bid_size;
        QQuantity ask_size;
        Timestamp timestamp;
        uint32_t venue_id;
        uint16_t flags;
    };

    Quantized quantize() const noexcept {
        return {
            symbol_id,
            static_cast<QPrice>(bid_price / PRICE_SCALE),
            static_cast<QPrice>(ask_price / PRICE_SCALE),
            static_cast<QQuantity>(std::min(bid_size, static_cast<Quantity>(32767))),
            static_cast<QQuantity>(std::min(ask_size, static_cast<Quantity>(32767))),
            timestamp,
            venue_id,
            flags
        };
    }
};

// Order types
enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    STOP = 2,
    STOP_LIMIT = 3,
    TRAILING_STOP = 4
};

enum class OrderSide : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderStatus : uint8_t {
    PENDING = 0,
    PARTIAL_FILL = 1,
    FILLED = 2,
    CANCELLED = 3,
    REJECTED = 4,
    EXPIRED = 5
};

enum class TimeInForce : uint8_t {
    DAY = 0,
    GTC = 1,        // Good Till Cancelled
    IOC = 2,        // Immediate Or Cancel
    FOK = 3,        // Fill Or Kill
    GTD = 4         // Good Till Date
};

// High-performance order structure
struct alignas(64) Order {
    OrderId order_id;
    SymbolId symbol_id;
    OrderType type;
    OrderSide side;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    OrderStatus status;
    Timestamp create_time;
    Timestamp update_time;
    VenueId venue_id;
    uint32_t account_id;
    TimeInForce tif;
    uint16_t flags;

    // Computed fields for performance
    Price avg_fill_price;  // Average fill price
    Quantity leaves_quantity;  // Remaining quantity

    void update_fill(Price fill_price, Quantity fill_qty) noexcept {
        filled_quantity += fill_qty;
        leaves_quantity = quantity - filled_quantity;

        // Update average fill price
        if (filled_quantity > 0) {
            avg_fill_price = ((avg_fill_price * (filled_quantity - fill_qty)) +
                            (fill_price * fill_qty)) / filled_quantity;
        } else {
            avg_fill_price = fill_price;
        }

        update_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        if (leaves_quantity == 0) {
            status = OrderStatus::FILLED;
        } else {
            status = OrderStatus::PARTIAL_FILL;
        }
    }
};

// Signal types
enum class SignalType : uint8_t {
    LONG = 1,
    SHORT = -1,
    NEUTRAL = 0,
    EXIT_LONG = -2,
    EXIT_SHORT = 2
};

// High-performance signal structure
struct alignas(32) Signal {
    SymbolId symbol_id;
    SignalType signal;
    float confidence;  // 0.0 to 1.0
    Timestamp timestamp;
    uint32_t strategy_id;
    Price target_price;  // Target entry/exit price
    Quantity target_quantity;

    // Quantized version for SIMD processing
    struct Quantized {
        SymbolId symbol_id;
        int8_t signal;  // -1, 0, 1
        uint8_t confidence;  // 0-255 (quantized from 0.0-1.0)
        Timestamp timestamp;
        uint32_t strategy_id;
        QPrice target_price;
        QQuantity target_quantity;
    };

    Quantized quantize() const noexcept {
        return {
            symbol_id,
            static_cast<int8_t>(signal),
            static_cast<uint8_t>(confidence * 255.0f),
            timestamp,
            strategy_id,
            static_cast<QPrice>(target_price / PRICE_SCALE),
            static_cast<QQuantity>(std::min(target_quantity, static_cast<Quantity>(32767)))
        };
    }
};

// Risk metrics structure
struct alignas(64) RiskMetrics {
    SymbolId symbol_id;
    Price current_price;
    Quantity position_size;
    Price entry_price;
    Price stop_loss_price;
    Price take_profit_price;
    float var_95;  // Value at Risk 95%
    float expected_shortfall;  // CVaR
    float sharpe_ratio;
    float max_drawdown;
    Timestamp timestamp;

    // Computed risk values
    Price unrealized_pnl;
    float position_var;
    bool breach_warning;

    void update_position(Price current_px) noexcept {
        current_price = current_px;
        unrealized_pnl = (current_px - entry_price) * position_size;
        position_var = var_95 * std::abs(position_size);
        timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    bool check_stop_loss() const noexcept {
        if (position_size > 0) {  // Long position
            return current_price <= stop_loss_price;
        } else {  // Short position
            return current_price >= stop_loss_price;
        }
    }

    bool check_take_profit() const noexcept {
        if (position_size > 0) {  // Long position
            return current_price >= take_profit_price;
        } else {  // Short position
            return current_price <= take_profit_price;
        }
    }
};

// Performance counters for monitoring
struct alignas(64) PerformanceCounters {
    std::atomic<uint64_t> signals_processed{0};
    std::atomic<uint64_t> orders_generated{0};
    std::atomic<uint64_t> orders_executed{0};
    std::atomic<uint64_t> risk_checks_performed{0};
    std::atomic<uint64_t> market_data_processed{0};

    // Latency measurements (in nanoseconds)
    std::atomic<uint64_t> avg_signal_latency{0};
    std::atomic<uint64_t> avg_order_latency{0};
    std::atomic<uint64_t> avg_risk_latency{0};
    std::atomic<uint64_t> max_signal_latency{0};
    std::atomic<uint64_t> max_order_latency{0};
    std::atomic<uint64_t> max_risk_latency{0};

    // Error counters
    std::atomic<uint64_t> signal_errors{0};
    std::atomic<uint64_t> order_errors{0};
    std::atomic<uint64_t> risk_errors{0};
    std::atomic<uint64_t> data_errors{0};

    void update_signal_latency(uint64_t latency_ns) noexcept {
        avg_signal_latency = (avg_signal_latency * 0.99) + (latency_ns * 0.01);
        if (latency_ns > max_signal_latency) {
            max_signal_latency = latency_ns;
        }
    }

    void update_order_latency(uint64_t latency_ns) noexcept {
        avg_order_latency = (avg_order_latency * 0.99) + (latency_ns * 0.01);
        if (latency_ns > max_order_latency) {
            max_order_latency = latency_ns;
        }
    }

    void update_risk_latency(uint64_t latency_ns) noexcept {
        avg_risk_latency = (avg_risk_latency * 0.99) + (latency_ns * 0.01);
        if (latency_ns > max_risk_latency) {
            max_risk_latency = latency_ns;
        }
    }
};

// Constants
constexpr size_t MAX_SYMBOLS = 10000;
constexpr size_t MAX_ORDERS = 1000000;
constexpr size_t MAX_SIGNALS = 100000;
constexpr size_t CACHE_LINE_SIZE = 64;

// Utility functions
inline Price price_from_double(double price) noexcept {
    return static_cast<Price>(price * PRICE_SCALE);
}

inline double price_to_double(Price price) noexcept {
    return static_cast<double>(price) / PRICE_SCALE;
}

inline QPrice quantize_price(Price price) noexcept {
    return static_cast<QPrice>(price / PRICE_SCALE);
}

inline Price dequantize_price(QPrice qprice) noexcept {
    return static_cast<Price>(qprice) * PRICE_SCALE;
}

inline Timestamp current_timestamp() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// SIMD helper functions (AVX-512 when available)
#ifdef __AVX512F__
#include <immintrin.h>

inline __m512d load_prices(const Price* prices) noexcept {
    // Convert fixed-point to double for SIMD processing
    double temp[8];
    for (int i = 0; i < 8; ++i) {
        temp[i] = price_to_double(prices[i]);
    }
    return _mm512_loadu_pd(temp);
}

inline void store_prices(Price* prices, __m512d values) noexcept {
    double temp[8];
    _mm512_storeu_pd(temp, values);
    for (int i = 0; i < 8; ++i) {
        prices[i] = price_from_double(temp[i]);
    }
}
#endif

} // namespace trading
