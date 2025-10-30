use crate::core::*;
use std::collections::HashMap;

/// Portfolio position management
pub struct PositionManager {
    positions: HashMap<SymbolId, Position>,
}

#[derive(Clone, Debug)]
pub struct Position {
    pub symbol_id: SymbolId,
    pub quantity: Quantity,
    pub avg_price: Price,
    pub unrealized_pnl: Price,
    pub realized_pnl: Price,
    pub last_update: Timestamp,
}

impl PositionManager {
    /// Create new position manager
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
        }
    }

    /// Update position with new trade
    pub fn update_position(&mut self, symbol_id: SymbolId, quantity: Quantity, price: Price) {
        let position = self.positions.entry(symbol_id).or_insert(Position {
            symbol_id,
            quantity: Decimal::ZERO,
            avg_price: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            last_update: current_timestamp(),
        });

        // Update average price using volume-weighted average
        let total_value = position.avg_price * position.quantity + price * quantity;
        let total_quantity = position.quantity + quantity;

        if total_quantity != Decimal::ZERO {
            position.avg_price = total_value / total_quantity;
        }

        position.quantity = total_quantity;
        position.last_update = current_timestamp();
    }

    /// Get position for symbol
    pub fn get_position(&self, symbol_id: SymbolId) -> Option<&Position> {
        self.positions.get(&symbol_id)
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<&Position> {
        self.positions.values().collect()
    }

    /// Calculate portfolio exposure
    pub fn portfolio_exposure(&self, current_prices: &HashMap<SymbolId, Price>) -> Price {
        self.positions.values()
            .map(|pos| {
                let current_price = current_prices.get(&pos.symbol_id).copied().unwrap_or(pos.avg_price);
                pos.quantity * current_price
            })
            .sum()
    }

    /// Calculate total unrealized P&L
    pub fn total_unrealized_pnl(&self, current_prices: &HashMap<SymbolId, Price>) -> Price {
        self.positions.values()
            .map(|pos| {
                let current_price = current_prices.get(&pos.symbol_id).copied().unwrap_or(pos.avg_price);
                let pnl = (current_price - pos.avg_price) * pos.quantity;
                pnl
            })
            .sum()
    }

    /// Clear all positions
    pub fn clear_positions(&mut self) {
        self.positions.clear();
    }
}
