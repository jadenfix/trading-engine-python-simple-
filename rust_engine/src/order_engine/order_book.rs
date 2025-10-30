use crate::core::*;
use std::collections::{BTreeMap, HashMap};

/// High-performance order book implementation
pub struct OrderBook {
    bids: BTreeMap<Price, Vec<OrderId>>,  // Price -> Order IDs (sorted descending)
    asks: BTreeMap<Price, Vec<OrderId>>,  // Price -> Order IDs (sorted ascending)
    orders: HashMap<OrderId, Order>,
}

#[derive(Clone, Debug)]
pub struct OrderBookSnapshot {
    pub symbol_id: SymbolId,
    pub best_bid: Price,
    pub best_ask: Price,
    pub bid_volume: Quantity,
    pub ask_volume: Quantity,
    pub spread: Price,
    pub timestamp: Timestamp,
}

impl OrderBook {
    /// Create new order book
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
        }
    }

    /// Add limit order to book
    pub fn add_limit_order(&mut self, order: Order) {
        let price_level = match order.side {
            OrderSide::Buy => self.bids.entry(order.price).or_insert_with(Vec::new),
            OrderSide::Sell => self.asks.entry(order.price).or_insert_with(Vec::new),
        };

        price_level.push(order.order_id);
        self.orders.insert(order.order_id, order);
    }

    /// Remove order from book
    pub fn remove_order(&mut self, order_id: OrderId) {
        if let Some(order) = self.orders.remove(&order_id) {
            let price_level = match order.side {
                OrderSide::Buy => self.bids.get_mut(&order.price),
                OrderSide::Sell => self.asks.get_mut(&order.price),
            };

            if let Some(level) = price_level {
                level.retain(|&id| id != order_id);
                if level.is_empty() {
                    match order.side {
                        OrderSide::Buy => { self.bids.remove(&order.price); }
                        OrderSide::Sell => { self.asks.remove(&order.price); }
                    }
                }
            }
        }
    }

    /// Modify existing order
    pub fn modify_order(&mut self, order_id: OrderId, new_price: Price, new_quantity: Quantity) {
        if let Some(order) = self.orders.get_mut(&order_id) {
            // Remove from old price level
            self.remove_order(order_id);

            // Update order
            order.price = new_price;
            order.quantity = new_quantity;

            // Re-add to new price level
            self.add_limit_order(order.clone());
        }
    }

    /// Get order book snapshot
    pub fn get_snapshot(&self, symbol_id: SymbolId) -> Option<OrderBookSnapshot> {
        let best_bid = self.bids.keys().next_back().copied()?;
        let best_ask = self.asks.keys().next().copied()?;

        let bid_volume = self.bids.get(&best_bid)
            .map(|orders| orders.len() as u64)
            .unwrap_or(0);
        let ask_volume = self.asks.get(&best_ask)
            .map(|orders| orders.len() as u64)
            .unwrap_or(0);

        Some(OrderBookSnapshot {
            symbol_id,
            best_bid,
            best_ask,
            bid_volume: Decimal::from(bid_volume),
            ask_volume: Decimal::from(ask_volume),
            spread: best_ask - best_bid,
            timestamp: current_timestamp(),
        })
    }

    /// Match orders (simplified implementation)
    pub fn match_orders(&mut self, _symbol_id: SymbolId) -> Vec<(OrderId, OrderId, Price, Quantity)> {
        // Simplified matching logic - in a real implementation this would be much more complex
        let mut matches = Vec::new();

        // This is a placeholder for proper order matching logic
        // In a real implementation, you would:
        // 1. Check if best bid >= best ask
        // 2. Match orders at the crossing price
        // 3. Update order statuses and quantities
        // 4. Remove filled orders

        matches
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.keys().next_back().copied()
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.keys().next().copied()
    }
}
