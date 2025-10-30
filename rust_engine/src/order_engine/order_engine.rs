use crate::core::*;
use std::collections::HashMap;

/// High-performance order management engine
pub struct OrderEngine {
    active_orders: HashMap<OrderId, Order>,
    order_book: order_book::OrderBook,
    order_router: order_router::OrderRouter,
    next_order_id: OrderId,
    performance_counters: PerformanceCounters,
}

impl OrderEngine {
    /// Create new order engine
    pub fn new() -> Self {
        Self {
            active_orders: HashMap::new(),
            order_book: order_book::OrderBook::new(),
            order_router: order_router::OrderRouter::new(),
            next_order_id: 1,
            performance_counters: PerformanceCounters::default(),
        }
    }

    /// Submit new order
    pub fn submit_order(&mut self, mut order: Order) -> OrderId {
        let start_time = std::time::Instant::now();

        // Assign order ID
        order.order_id = self.next_order_id;
        self.next_order_id += 1;

        // Set timestamps
        let now = current_timestamp();
        order.create_time = now;
        order.update_time = now;

        let order_id = order.order_id;

        // Add to active orders
        self.active_orders.insert(order_id, order.clone());

        // Route order
        self.order_router.route_order(&order);

        // Add to order book if limit order
        if matches!(order.order_type, OrderType::Limit) {
            self.order_book.add_limit_order(order);
        }

        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.performance_counters.update_order_latency(processing_time);
        self.performance_counters.orders_submitted += 1;

        order_id
    }

    /// Cancel order
    pub fn cancel_order(&mut self, order_id: OrderId) -> bool {
        if let Some(mut order) = self.active_orders.remove(&order_id) {
            order.status = OrderStatus::Cancelled;
            order.update_time = current_timestamp();
            self.performance_counters.orders_cancelled += 1;
            true
        } else {
            false
        }
    }

    /// Modify order
    pub fn modify_order(&mut self, order_id: OrderId, new_price: Price, new_quantity: Quantity) -> bool {
        if let Some(order) = self.active_orders.get_mut(&order_id) {
            if matches!(order.status, OrderStatus::Pending) {
                order.price = new_price;
                order.quantity = new_quantity;
                order.update_time = current_timestamp();

                // Update order book
                self.order_book.modify_order(order_id, new_price, new_quantity);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get order status
    pub fn get_order(&self, order_id: OrderId) -> Option<&Order> {
        self.active_orders.get(&order_id)
    }

    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<&Order> {
        self.active_orders.values().collect()
    }

    /// Get order book snapshot
    pub fn get_order_book_snapshot(&self, symbol_id: SymbolId) -> Option<order_book::OrderBookSnapshot> {
        self.order_book.get_snapshot(symbol_id)
    }

    /// Match orders (simplified)
    pub fn match_orders(&mut self, symbol_id: SymbolId) -> Vec<(OrderId, OrderId, Price, Quantity)> {
        self.order_book.match_orders(symbol_id)
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceCounters {
        &self.performance_counters
    }

    /// Reset performance counters
    pub fn reset_performance_counters(&mut self) {
        self.performance_counters = PerformanceCounters::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;

    #[test]
    fn test_order_engine_creation() {
        let engine = OrderEngine::new();
        let stats = engine.performance_stats();
        assert_eq!(stats.orders_submitted, 0);
    }

    #[test]
    fn test_order_submission() {
        let mut engine = OrderEngine::new();

        let order = Order {
            symbol_id: 1,
            order_type: OrderType::Limit,
            side: OrderSide::Buy,
            price: price_from_float(100.0),
            quantity: Decimal::from(100),
            ..Default::default()
        };

        let order_id = engine.submit_order(order);
        assert!(order_id > 0);

        // Check order was added
        let retrieved_order = engine.get_order(order_id).unwrap();
        assert_eq!(retrieved_order.symbol_id, 1);
        assert_eq!(retrieved_order.order_type, OrderType::Limit);
    }

    #[test]
    fn test_order_cancellation() {
        let mut engine = OrderEngine::new();

        let order = Order {
            symbol_id: 1,
            order_type: OrderType::Market,
            side: OrderSide::Sell,
            quantity: Decimal::from(50),
            ..Default::default()
        };

        let order_id = engine.submit_order(order);
        assert!(engine.get_order(order_id).is_some());

        // Cancel order
        let cancelled = engine.cancel_order(order_id);
        assert!(cancelled);

        // Check order status
        let cancelled_order = engine.get_order(order_id).unwrap();
        assert_eq!(cancelled_order.status, OrderStatus::Cancelled);
    }
}
