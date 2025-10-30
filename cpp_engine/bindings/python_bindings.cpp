#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/signal_engine/signal_engine.hpp"
#include "../include/risk_engine/risk_engine.hpp"
#include "../include/order_engine/order_engine.hpp"
#include "../include/core/types.hpp"

namespace py = pybind11;
using namespace trading;

// Python bindings for the C++ trading engine components
PYBIND11_MODULE(python_bindings, m) {
    m.doc() = "High-performance C++ trading engine bindings for Python";

    // Core types bindings
    py::class_<MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("symbol_id", &MarketData::symbol_id)
        .def_readwrite("bid_price", &MarketData::bid_price)
        .def_readwrite("ask_price", &MarketData::ask_price)
        .def_readwrite("bid_size", &MarketData::bid_size)
        .def_readwrite("ask_size", &MarketData::ask_size)
        .def_readwrite("timestamp", &MarketData::timestamp)
        .def_readwrite("venue_id", &MarketData::venue_id)
        .def_readwrite("flags", &MarketData::flags)
        .def("quantize", &MarketData::quantize);

    py::class_<Signal>(m, "Signal")
        .def(py::init<>())
        .def_readwrite("symbol_id", &Signal::symbol_id)
        .def_readwrite("signal", &Signal::signal)
        .def_readwrite("confidence", &Signal::confidence)
        .def_readwrite("timestamp", &Signal::timestamp)
        .def_readwrite("strategy_id", &Signal::strategy_id)
        .def_readwrite("target_price", &Signal::target_price)
        .def_readwrite("target_quantity", &Signal::target_quantity)
        .def("quantize", &Signal::quantize);

    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id", &Order::order_id)
        .def_readwrite("symbol_id", &Order::symbol_id)
        .def_readwrite("type", &Order::type)
        .def_readwrite("side", &Order::side)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("status", &Order::status)
        .def_readwrite("create_time", &Order::create_time)
        .def_readwrite("update_time", &Order::update_time)
        .def_readwrite("venue_id", &Order::venue_id)
        .def_readwrite("tif", &Order::tif)
        .def("update_fill", &Order::update_fill);

    py::class_<RiskMetrics>(m, "RiskMetrics")
        .def(py::init<>())
        .def_readwrite("symbol_id", &RiskMetrics::symbol_id)
        .def_readwrite("current_price", &RiskMetrics::current_price)
        .def_readwrite("position_size", &RiskMetrics::position_size)
        .def_readwrite("entry_price", &RiskMetrics::entry_price)
        .def_readwrite("stop_loss_price", &RiskMetrics::stop_loss_price)
        .def_readwrite("take_profit_price", &RiskMetrics::take_profit_price)
        .def_readwrite("var_95", &RiskMetrics::var_95)
        .def_readwrite("expected_shortfall", &RiskMetrics::expected_shortfall)
        .def_readwrite("sharpe_ratio", &RiskMetrics::sharpe_ratio)
        .def_readwrite("max_drawdown", &RiskMetrics::max_drawdown)
        .def_readwrite("timestamp", &RiskMetrics::timestamp)
        .def("update_position", &RiskMetrics::update_position)
        .def("check_stop_loss", &RiskMetrics::check_stop_loss)
        .def("check_take_profit", &RiskMetrics::check_take_profit);

    py::class_<PerformanceCounters>(m, "PerformanceCounters")
        .def(py::init<>())
        .def_readwrite("signals_processed", &PerformanceCounters::signals_processed)
        .def_readwrite("orders_submitted", &PerformanceCounters::orders_submitted)
        .def_readwrite("orders_executed", &PerformanceCounters::orders_executed)
        .def_readwrite("risk_checks_performed", &PerformanceCounters::risk_checks_performed)
        .def_readwrite("market_data_processed", &PerformanceCounters::market_data_processed)
        .def_readwrite("avg_signal_latency", &PerformanceCounters::avg_signal_latency)
        .def_readwrite("avg_order_latency", &PerformanceCounters::avg_order_latency)
        .def_readwrite("avg_risk_latency", &PerformanceCounters::avg_risk_latency)
        .def_readwrite("max_signal_latency", &PerformanceCounters::max_signal_latency)
        .def_readwrite("max_order_latency", &PerformanceCounters::max_order_latency)
        .def_readwrite("max_risk_latency", &PerformanceCounters::max_risk_latency)
        .def_readwrite("signal_errors", &PerformanceCounters::signal_errors)
        .def_readwrite("order_errors", &PerformanceCounters::order_errors)
        .def_readwrite("risk_errors", &PerformanceCounters::risk_errors)
        .def_readwrite("data_errors", &PerformanceCounters::data_errors)
        .def("update_signal_latency", &PerformanceCounters::update_signal_latency)
        .def("update_order_latency", &PerformanceCounters::update_order_latency)
        .def("update_risk_latency", &PerformanceCounters::update_risk_latency);

    // Signal Engine bindings
    py::class_<SignalEngine>(m, "SignalEngine")
        .def(py::init<>())
        .def("generate_signals", &SignalEngine::generate_signals)
        .def("generate_signals_simd", &SignalEngine::generate_signals_simd)
        .def("generate_momentum_signals", &SignalEngine::generate_momentum_signals)
        .def("generate_mean_reversion_signals", &SignalEngine::generate_mean_reversion_signals)
        .def("generate_volatility_signals", &SignalEngine::generate_volatility_signals)
        .def("generate_arbitrage_signals", &SignalEngine::generate_arbitrage_signals)
        .def("combine_signals", &SignalEngine::combine_signals)
        .def("filter_signals", &SignalEngine::filter_signals)
        .def("update_signal_confidence", &SignalEngine::update_signal_confidence)
        .def("normalize_signal_strength", &SignalEngine::normalize_signal_strength)
        .def("get_performance_counters", &SignalEngine::get_performance_counters)
        .def("reset_performance_counters", &SignalEngine::reset_performance_counters);

    // Risk Engine bindings
    py::class_<RiskEngine>(m, "RiskEngine")
        .def(py::init<>())
        .def("calculate_portfolio_risk", &RiskEngine::calculate_portfolio_risk)
        .def("check_risk_limits", &RiskEngine::check_risk_limits)
        .def("calculate_position_risk", &RiskEngine::calculate_position_risk)
        .def("calculate_portfolio_risk_metrics", &RiskEngine::calculate_portfolio_risk_metrics)
        .def("update_risk_metrics", &RiskEngine::update_risk_metrics)
        .def("set_risk_limits", &RiskEngine::set_risk_limits)
        .def("get_risk_limits", &RiskEngine::get_risk_limits)
        .def("run_stress_tests", &RiskEngine::run_stress_tests)
        .def("get_performance_counters", &RiskEngine::get_performance_counters)
        .def("reset_performance_counters", &RiskEngine::reset_performance_counters);

    // Risk Report bindings
    py::class_<RiskEngine::RiskReport>(m, "RiskReport")
        .def(py::init<>())
        .def_readwrite("total_var_95", &RiskEngine::RiskReport::total_var_95)
        .def_readwrite("total_cvar_95", &RiskEngine::RiskReport::total_cvar_95)
        .def_readwrite("portfolio_volatility", &RiskEngine::RiskReport::portfolio_volatility)
        .def_readwrite("max_drawdown", &RiskEngine::RiskReport::max_drawdown)
        .def_readwrite("positions_at_risk", &RiskEngine::RiskReport::positions_at_risk)
        .def_readwrite("risk_warnings", &RiskEngine::RiskReport::risk_warnings)
        .def_readwrite("generated_at", &RiskEngine::RiskReport::generated_at);

    // Bind RiskEngine::generate_risk_report separately
    py::class_<RiskEngine> risk_engine_class(m, "RiskEngine");
    risk_engine_class.def("generate_risk_report", &RiskEngine::generate_risk_report);

    // Order Engine bindings
    py::class_<OrderEngine>(m, "OrderEngine")
        .def(py::init<>())
        .def("submit_order", &OrderEngine::submit_order)
        .def("cancel_order", &OrderEngine::cancel_order)
        .def("modify_order", &OrderEngine::modify_order)
        .def("submit_orders", &OrderEngine::submit_orders)
        .def("cancel_all_orders", &OrderEngine::cancel_all_orders)
        .def("get_order_status", &OrderEngine::get_order_status)
        .def("get_pending_orders", &OrderEngine::get_pending_orders)
        .def("get_filled_orders", &OrderEngine::get_filled_orders)
        .def("get_execution_reports", &OrderEngine::get_execution_reports)
        .def("select_optimal_venue", &OrderEngine::select_optimal_venue)
        .def("set_routing_strategy", &OrderEngine::set_routing_strategy)
        .def("estimate_market_impact", &OrderEngine::estimate_market_impact)
        .def("process_order_queue", &OrderEngine::process_order_queue)
        .def("get_queue_depth", &OrderEngine::get_queue_depth)
        .def("get_statistics", &OrderEngine::get_statistics)
        .def("reset_statistics", &OrderEngine::reset_statistics);

    // Order Engine enums
    py::enum_<OrderEngine::RoutingStrategy>(m, "RoutingStrategy")
        .value("PRICE_PRIORITY", OrderEngine::RoutingStrategy::PRICE_PRIORITY)
        .value("SIZE_PRIORITY", OrderEngine::RoutingStrategy::SIZE_PRIORITY)
        .value("SPEED_PRIORITY", OrderEngine::RoutingStrategy::SPEED_PRIORITY)
        .value("COST_PRIORITY", OrderEngine::RoutingStrategy::COST_PRIORITY)
        .value("SMART_ROUTING", OrderEngine::RoutingStrategy::SMART_ROUTING)
        .value("VWAP", OrderEngine::RoutingStrategy::VWAP)
        .value("TWAP", OrderEngine::RoutingStrategy::TWAP)
        .value("ICEBERG", OrderEngine::RoutingStrategy::ICEBERG)
        .value("ADAPTIVE", OrderEngine::RoutingStrategy::ADAPTIVE);

    // Order Engine structs
    py::class_<OrderEngine::ExecutionReport>(m, "ExecutionReport")
        .def(py::init<>())
        .def_readwrite("order_id", &OrderEngine::ExecutionReport::order_id)
        .def_readwrite("execution_time", &OrderEngine::ExecutionReport::execution_time)
        .def_readwrite("execution_price", &OrderEngine::ExecutionReport::execution_price)
        .def_readwrite("execution_quantity", &OrderEngine::ExecutionReport::execution_quantity)
        .def_readwrite("venue_id", &OrderEngine::ExecutionReport::venue_id)
        .def_readwrite("execution_id", &OrderEngine::ExecutionReport::execution_id)
        .def_readwrite("is_partial_fill", &OrderEngine::ExecutionReport::is_partial_fill);

    py::class_<OrderEngine::MarketImpact>(m, "MarketImpact")
        .def(py::init<>())
        .def_readwrite("expected_price_impact", &OrderEngine::MarketImpact::expected_price_impact)
        .def_readwrite("expected_fill_size", &OrderEngine::MarketImpact::expected_fill_size)
        .def_readwrite("expected_execution_time", &OrderEngine::MarketImpact::expected_execution_time)
        .def_readwrite("execution_probability", &OrderEngine::MarketImpact::execution_probability);

    py::class_<OrderEngine::OrderEngineStats>(m, "OrderEngineStats")
        .def(py::init<>())
        .def_readwrite("orders_submitted", &OrderEngine::OrderEngineStats::orders_submitted)
        .def_readwrite("orders_executed", &OrderEngine::OrderEngineStats::orders_executed)
        .def_readwrite("orders_cancelled", &OrderEngine::OrderEngineStats::orders_cancelled)
        .def_readwrite("orders_rejected", &OrderEngine::OrderEngineStats::orders_rejected)
        .def_readwrite("avg_execution_latency_ms", &OrderEngine::OrderEngineStats::avg_execution_latency_ms)
        .def_readwrite("avg_fill_latency_ms", &OrderEngine::OrderEngineStats::avg_fill_latency_ms)
        .def_readwrite("order_fill_rate", &OrderEngine::OrderEngineStats::order_fill_rate)
        .def_readwrite("venue_execution_counts", &OrderEngine::OrderEngineStats::venue_execution_counts);

    // Core utility functions
    m.def("price_from_double", &price_from_double, "Convert double to fixed-point price");
    m.def("price_to_double", &price_to_double, "Convert fixed-point price to double");
    m.def("quantize_price", &quantize_price, "Quantize price for fast computation");
    m.def("dequantize_price", &dequantize_price, "Dequantize price back to full precision");
    m.def("current_timestamp", &current_timestamp, "Get current timestamp in nanoseconds");

    // Enums
    py::enum_<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::MARKET)
        .value("LIMIT", OrderType::LIMIT)
        .value("STOP", OrderType::STOP)
        .value("STOP_LIMIT", OrderType::STOP_LIMIT)
        .value("TRAILING_STOP", OrderType::TRAILING_STOP);

    py::enum_<OrderSide>(m, "OrderSide")
        .value("BUY", OrderSide::BUY)
        .value("SELL", OrderSide::SELL);

    py::enum_<OrderStatus>(m, "OrderStatus")
        .value("PENDING", OrderStatus::PENDING)
        .value("PARTIAL_FILL", OrderStatus::PARTIAL_FILL)
        .value("FILLED", OrderStatus::FILLED)
        .value("CANCELLED", OrderStatus::CANCELLED)
        .value("REJECTED", OrderStatus::REJECTED)
        .value("EXPIRED", OrderStatus::EXPIRED);

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("DAY", TimeInForce::DAY)
        .value("GTC", TimeInForce::GTC)
        .value("IOC", TimeInForce::IOC)
        .value("FOK", TimeInForce::FOK)
        .value("GTD", TimeInForce::GTD);

    py::enum_<SignalType>(m, "SignalType")
        .value("LONG", SignalType::LONG)
        .value("SHORT", SignalType::SHORT)
        .value("NEUTRAL", SignalType::NEUTRAL)
        .value("EXIT_LONG", SignalType::EXIT_LONG)
        .value("EXIT_SHORT", SignalType::EXIT_SHORT);

    // Signal utilities
    py::class_<signal_utils::SignalQuality>(m, "SignalQuality")
        .def(py::init<>())
        .def_readwrite("consistency", &signal_utils::SignalQuality::consistency)
        .def_readwrite("reliability", &signal_utils::SignalQuality::reliability)
        .def_readwrite("timeliness", &signal_utils::SignalQuality::timeliness)
        .def_readwrite("strength", &signal_utils::SignalQuality::strength);

    m.def("assess_signal_quality", &signal_utils::assess_signal_quality,
          "Assess the quality of a trading signal");

    // Risk utilities
    m.def("correlation_coefficient", &risk_utils::correlation_coefficient,
          "Calculate correlation coefficient between two return series");

    m.def("portfolio_volatility", &risk_utils::portfolio_volatility,
          "Calculate portfolio volatility given weights and covariance matrix");

    m.def("calculate_beta", &risk_utils::calculate_beta,
          "Calculate beta relative to market returns");

    m.def("sortino_ratio", &risk_utils::sortino_ratio,
          "Calculate Sortino ratio for return series");

    // Numpy array support for performance
    py::class_<py::array_t<float>>(m, "FloatArray")
        .def(py::init([](py::array_t<float> arr) { return arr; }));

    // Vectorized operations for bulk processing
    m.def("vectorized_price_quantization", [](const std::vector<double>& prices) {
        std::vector<QPrice> result;
        result.reserve(prices.size());
        for (double price : prices) {
            result.push_back(PriceQuantizer::quantize(price_from_double(price)));
        }
        return result;
    }, "Vectorized price quantization for bulk processing");

    m.def("vectorized_signal_confidence_update", [](std::vector<Signal>& signals) {
        for (auto& signal : signals) {
            // Apply confidence decay based on age
            Timestamp age = current_timestamp() - signal.timestamp;
            float decay_factor = std::exp(-static_cast<double>(age) / 1e10);  // 10 second half-life
            signal.confidence *= decay_factor;

            // Normalize to [0, 1]
            signal.confidence = std::clamp(signal.confidence, 0.0f, 1.0f);
        }
    }, "Vectorized signal confidence updates");

    m.def("bulk_risk_calculation", [](const std::vector<Order>& positions,
                                     const std::vector<MarketData>& market_data) {
        std::vector<RiskMetrics> risks;
        risks.reserve(positions.size());

        for (const auto& position : positions) {
            RiskMetrics risk{};
            risk.symbol_id = position.symbol_id;
            risk.position_size = position.quantity;
            risk.entry_price = position.price;

            // Find matching market data
            for (const auto& data : market_data) {
                if (data.symbol_id == position.symbol_id) {
                    risk.update_position(data.ask_price);  // Use ask for risk calculation
                    break;
                }
            }

            risks.push_back(risk);
        }

        return risks;
    }, "Bulk risk calculation for multiple positions");

    // Version information
    m.attr("__version__") = "1.0.0";
    m.attr("BUILD_TYPE") = "RELEASE";
}
