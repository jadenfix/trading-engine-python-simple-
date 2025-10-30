#pragma once

#include "../core/types.hpp"
#include "../core/lock_free_queue.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <string>

namespace trading {

// Forward declarations
class OrderRouter;
class FIXProtocol;
class OrderBook;

class OrderEngine {
public:
    OrderEngine();
    ~OrderEngine() = default;

    // Disable copy/move for performance
    OrderEngine(const OrderEngine&) = delete;
    OrderEngine& operator=(const OrderEngine&) = delete;
    OrderEngine(OrderEngine&&) = delete;
    OrderEngine& operator=(OrderEngine&&) = delete;

    // Order submission and management
    OrderId submit_order(const Order& order);
    bool cancel_order(OrderId order_id);
    bool modify_order(OrderId order_id, Price new_price, Quantity new_quantity);

    // Bulk order operations
    std::vector<OrderId> submit_orders(const std::vector<Order>& orders);
    size_t cancel_all_orders(SymbolId symbol_id = 0);  // 0 = all symbols

    // Order status queries
    Order get_order_status(OrderId order_id) const noexcept;
    std::vector<Order> get_pending_orders(SymbolId symbol_id = 0) const noexcept;
    std::vector<Order> get_filled_orders(SymbolId symbol_id = 0) const noexcept;

    // Execution reporting
    struct ExecutionReport {
        OrderId order_id;
        Timestamp execution_time;
        Price execution_price;
        Quantity execution_quantity;
        VenueId venue_id;
        std::string execution_id;
        bool is_partial_fill;
    };

    std::vector<ExecutionReport> get_execution_reports(OrderId order_id) const noexcept;

    // Smart order routing
    VenueId select_optimal_venue(const Order& order,
                               const std::vector<MarketData>& market_data) noexcept;

    // Order routing strategies
    enum class RoutingStrategy {
        PRICE_PRIORITY,      // Route to best price
        SIZE_PRIORITY,       // Route to largest size
        SPEED_PRIORITY,      // Route to fastest execution
        COST_PRIORITY,       // Route to lowest cost
        SMART_ROUTING,       // Intelligent routing based on multiple factors
        VWAP,               // Volume weighted average price
        TWAP,               // Time weighted average price
        ICEBERG,            // Hidden order execution
        ADAPTIVE            // Adaptive routing based on market conditions
    };

    void set_routing_strategy(RoutingStrategy strategy) noexcept {
        routing_strategy_ = strategy;
    }

    // Market impact estimation
    struct MarketImpact {
        Price expected_price_impact;
        Quantity expected_fill_size;
        Timestamp expected_execution_time;
        float execution_probability;
    };

    MarketImpact estimate_market_impact(const Order& order,
                                      const MarketData& market_data) noexcept;

    // Order queue management
    void process_order_queue() noexcept;
    size_t get_queue_depth() const noexcept;

    // Performance monitoring
    struct OrderEngineStats {
        uint64_t orders_submitted;
        uint64_t orders_executed;
        uint64_t orders_cancelled;
        uint64_t orders_rejected;
        double avg_execution_latency_ms;
        double avg_fill_latency_ms;
        double order_fill_rate;
        std::unordered_map<VenueId, uint64_t> venue_execution_counts;
    };

    OrderEngineStats get_statistics() const noexcept;
    void reset_statistics() noexcept;

private:
    // Core components
    std::unique_ptr<OrderRouter> order_router_;
    std::unique_ptr<FIXProtocol> fix_protocol_;
    std::unique_ptr<OrderBook> order_book_;

    // Order storage and management
    std::unordered_map<OrderId, Order> active_orders_;
    std::unordered_map<OrderId, std::vector<ExecutionReport>> execution_reports_;
    LockFreeQueue<Order> order_queue_;
    LockFreeQueue<ExecutionReport> execution_queue_;

    // Order ID generation (thread-safe)
    std::atomic<OrderId> next_order_id_{1};

    // Configuration
    RoutingStrategy routing_strategy_;
    struct OrderEngineConfig {
        size_t max_orders_per_second = 10000;
        size_t max_queue_depth = 100000;
        Timestamp max_order_age_ns = 30000000000;  // 30 seconds
        bool enable_smart_routing = true;
        bool enable_market_impact_estimation = true;
        float max_market_impact_threshold = 0.001f;  // 0.1%
    } config_;

    // Internal methods
    OrderId generate_order_id() noexcept;
    bool validate_order(const Order& order) const noexcept;
    void route_order(Order order) noexcept;
    void process_execution_report(const ExecutionReport& report) noexcept;
    void update_order_status(OrderId order_id, OrderStatus new_status) noexcept;

    // Smart routing algorithms
    VenueId route_price_priority(const Order& order,
                               const std::vector<MarketData>& market_data) noexcept;
    VenueId route_size_priority(const Order& order,
                              const std::vector<MarketData>& market_data) noexcept;
    VenueId route_cost_priority(const Order& order,
                              const std::vector<MarketData>& market_data) noexcept;
};

// Order Router - Intelligent order routing across venues
class OrderRouter {
public:
    OrderRouter();
    ~OrderRouter() = default;

    // Venue management
    void add_venue(VenueId venue_id, const std::string& venue_name,
                  const std::string& connection_string);
    void remove_venue(VenueId venue_id);
    bool is_venue_active(VenueId venue_id) const noexcept;

    // Routing decisions
    VenueId route_order(const Order& order,
                       const std::vector<MarketData>& market_data,
                       OrderEngine::RoutingStrategy strategy) noexcept;

    // Venue performance tracking
    struct VenuePerformance {
        uint64_t orders_routed;
        uint64_t orders_executed;
        uint64_t orders_rejected;
        double avg_execution_time_ms;
        double avg_fill_rate;
        double avg_price_improvement_bps;
        Timestamp last_execution_time;
        bool is_active;
    };

    VenuePerformance get_venue_performance(VenueId venue_id) const noexcept;
    std::vector<std::pair<VenueId, VenuePerformance>> get_all_venue_performance() const noexcept;

    // Venue selection algorithms
    VenueId select_by_price(const Order& order, const std::vector<MarketData>& market_data) noexcept;
    VenueId select_by_size(const Order& order, const std::vector<MarketData>& market_data) noexcept;
    VenueId select_by_speed(const Order& order, const std::vector<MarketData>& market_data) noexcept;

private:
    // Venue information
    struct VenueInfo {
        VenueId venue_id;
        std::string name;
        std::string connection_string;
        VenuePerformance performance;
        std::atomic<bool> is_active{true};
    };

    std::unordered_map<VenueId, VenueInfo> venues_;

    // Routing helpers
    std::vector<VenueId> get_active_venues() const noexcept;
    double calculate_venue_score(VenueId venue_id, const Order& order,
                               const std::vector<MarketData>& market_data) noexcept;
};

// FIX Protocol Handler - High-performance FIX protocol implementation
class FIXProtocol {
public:
    FIXProtocol();
    ~FIXProtocol() = default;

    // FIX message types
    enum class FIXMessageType {
        NEW_ORDER = 'D',
        CANCEL_ORDER = 'F',
        MODIFY_ORDER = 'G',
        ORDER_STATUS_REQUEST = 'H',
        EXECUTION_REPORT = '8',
        ORDER_CANCEL_REJECT = '9',
        LOGON = 'A',
        LOGOUT = '5',
        HEARTBEAT = '0',
        TEST_REQUEST = '1'
    };

    // Message creation
    std::string create_new_order_message(const Order& order, VenueId venue_id) noexcept;
    std::string create_cancel_message(OrderId order_id, VenueId venue_id) noexcept;
    std::string create_modify_message(OrderId order_id, Price new_price,
                                    Quantity new_quantity, VenueId venue_id) noexcept;

    // Message parsing
    ExecutionReport parse_execution_report(const std::string& fix_message) noexcept;
    Order parse_order_acknowledgment(const std::string& fix_message) noexcept;

    // Connection management
    bool connect_to_venue(VenueId venue_id, const std::string& host, int port);
    void disconnect_from_venue(VenueId venue_id);
    bool is_connected(VenueId venue_id) const noexcept;

    // Message sending/receiving
    bool send_message(VenueId venue_id, const std::string& message) noexcept;
    std::string receive_message(VenueId venue_id, Timestamp timeout_ns = 1000000000) noexcept;

private:
    // FIX protocol constants
    static constexpr char SOH = '\x01';
    static constexpr int MAX_FIX_MESSAGE_SIZE = 4096;

    // Message sequence numbers (per venue)
    std::unordered_map<VenueId, std::atomic<uint32_t>> sequence_numbers_;

    // Connection management
    struct VenueConnection {
        VenueId venue_id;
        int socket_fd;
        std::string host;
        int port;
        bool is_connected;
        Timestamp last_message_time;
        std::atomic<uint32_t> heartbeat_interval_ms{30000};
    };

    std::unordered_map<VenueId, VenueConnection> connections_;

    // FIX message building helpers
    std::string build_fix_header(FIXMessageType msg_type, uint32_t sequence_number) noexcept;
    std::string build_fix_trailer(const std::string& message_body) noexcept;
    uint32_t calculate_checksum(const std::string& message) noexcept;

    // Message parsing helpers
    std::unordered_map<std::string, std::string> parse_fix_fields(const std::string& message) noexcept;
    FIXMessageType get_message_type(const std::string& message) noexcept;
};

// Order Book - High-performance order book management
class OrderBook {
public:
    OrderBook(SymbolId symbol_id);
    ~OrderBook() = default;

    // Order book operations
    void add_bid(Price price, Quantity quantity, OrderId order_id) noexcept;
    void add_ask(Price price, Quantity quantity, OrderId order_id) noexcept;
    void remove_bid(Price price, OrderId order_id) noexcept;
    void remove_ask(Price price, Quantity quantity, OrderId order_id) noexcept;

    // Order book queries
    Price get_best_bid() const noexcept;
    Price get_best_ask() const noexcept;
    Quantity get_bid_size_at_price(Price price) const noexcept;
    Quantity get_ask_size_at_price(Price price) const noexcept;

    struct OrderBookSnapshot {
        Price best_bid;
        Price best_ask;
        Quantity bid_size;
        Quantity ask_size;
        Price mid_price;
        Price spread;
        Quantity total_bid_depth;
        Quantity total_ask_depth;
        size_t bid_levels;
        size_t ask_levels;
        Timestamp snapshot_time;
    };

    OrderBookSnapshot get_snapshot() const noexcept;

    // Market impact estimation
    OrderEngine::MarketImpact estimate_market_impact(const Order& order) const noexcept;

    // Liquidity analysis
    struct LiquidityMetrics {
        Quantity bid_liquidity_depth;
        Quantity ask_liquidity_depth;
        float bid_ask_spread_ratio;
        float market_resiliency_score;
        Timestamp calculated_at;
    };

    LiquidityMetrics calculate_liquidity_metrics() const noexcept;

    // Price discovery
    Price estimate_fair_value() const noexcept;
    Price estimate_vwap(int lookback_seconds) const noexcept;

private:
    SymbolId symbol_id_;

    // Order book storage - using std::map for price-time priority
    std::map<Price, std::vector<std::pair<OrderId, Quantity>>, std::greater<Price>> bid_book_;
    std::map<Price, std::vector<std::pair<OrderId, Quantity>>> ask_book_;

    // Performance tracking
    std::atomic<uint64_t> updates_processed_{0};
    std::atomic<uint64_t> queries_processed_{0};

    // Helper methods
    Quantity get_total_depth(const decltype(bid_book_)& book) const noexcept;
    size_t get_level_count(const decltype(bid_book_)& book) const noexcept;
    void update_price_levels(Price price, Quantity quantity,
                           decltype(bid_book_)& book, bool is_add) noexcept;
};

// Utility functions for order management
namespace order_utils {

// Order validation
bool validate_order(const Order& order) noexcept;
bool validate_order_modification(const Order& original, Price new_price, Quantity new_quantity) noexcept;

// Order cost calculation
struct OrderCosts {
    Price commission;
    Price exchange_fee;
    Price liquidity_fee;
    Price total_cost;
};

OrderCosts calculate_order_costs(const Order& order, VenueId venue_id) noexcept;

// Order execution quality metrics
struct ExecutionQuality {
    float price_improvement_bps;
    float market_impact_bps;
    float execution_speed_seconds;
    float fill_rate_percentage;
    float slippage_bps;
};

ExecutionQuality calculate_execution_quality(const Order& original_order,
                                          const std::vector<Order::Execution>& executions) noexcept;

// Smart order splitting algorithms
std::vector<Order> split_order_for_twap(const Order& original_order,
                                      Timestamp execution_window_ns,
                                      size_t num_slices) noexcept;

std::vector<Order> split_order_for_vwap(const Order& original_order,
                                      const std::vector<MarketData>& expected_volume_profile,
                                      size_t num_slices) noexcept;

// Iceberg order management
class IcebergOrder {
private:
    Order parent_order_;
    Quantity peak_size_;  // Visible size
    std::vector<Order> child_orders_;
    size_t current_child_index_;

public:
    IcebergOrder(const Order& parent_order, Quantity peak_size);
    std::vector<Order> get_next_orders(Quantity available_liquidity) noexcept;
    bool is_complete() const noexcept;
    Order get_parent_order() const noexcept;
};

} // namespace order_utils

} // namespace trading
