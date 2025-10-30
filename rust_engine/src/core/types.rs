use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use std::time::{SystemTime, UNIX_EPOCH};

/// Financial types with high precision
pub type Price = Decimal;        // High-precision price representation
pub type Quantity = Decimal;     // High-precision quantity
pub type Timestamp = u64;        // Nanoseconds since epoch
pub type OrderId = u64;          // Unique order identifier
pub type SymbolId = u32;         // Internal symbol identifier
pub type VenueId = u16;          // Trading venue identifier

/// Quantized types for ultra-fast computation
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct QPrice(pub i32);      // Quantized price (precision: 0.01)

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct QQuantity(pub i16);   // Quantized quantity (precision: 1)

/// High-performance market data structure
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct MarketData {
    pub symbol_id: SymbolId,
    pub bid_price: Price,
    pub ask_price: Price,
    pub bid_size: Quantity,
    pub ask_size: Quantity,
    pub timestamp: Timestamp,
    pub venue_id: VenueId,
    pub flags: u16,
}

impl MarketData {
    /// Create quantized version for SIMD processing
    pub fn quantize(&self) -> MarketDataQuantized {
        MarketDataQuantized {
            symbol_id: self.symbol_id,
            bid_price: QPrice::from_price(self.bid_price),
            ask_price: QPrice::from_price(self.ask_price),
            bid_size: QQuantity::from_quantity(self.bid_size),
            ask_size: QQuantity::from_quantity(self.ask_size),
            timestamp: self.timestamp,
            venue_id: self.venue_id,
            flags: self.flags,
        }
    }
}

/// Quantized market data for SIMD processing
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct MarketDataQuantized {
    pub symbol_id: SymbolId,
    pub bid_price: QPrice,
    pub ask_price: QPrice,
    pub bid_size: QQuantity,
    pub ask_size: QQuantity,
    pub timestamp: Timestamp,
    pub venue_id: VenueId,
    pub flags: u16,
}

impl QPrice {
    /// Convert Price to QPrice (precision: 0.01)
    pub fn from_price(price: Price) -> Self {
        let scaled = price * Decimal::new(100, 0);
        let quantized = scaled.round().to_i32().unwrap_or(0);
        QPrice(quantized)
    }

    /// Convert QPrice back to Price
    pub fn to_price(self) -> Price {
        Decimal::new(self.0 as i64, 2) // Divide by 100
    }

    /// Fast arithmetic operations
    pub fn add(self, other: QPrice) -> QPrice {
        QPrice(self.0 + other.0)
    }

    pub fn subtract(self, other: QPrice) -> QPrice {
        QPrice(self.0 - other.0)
    }

    pub fn multiply(self, other: QPrice) -> QPrice {
        // Scale down to prevent overflow
        QPrice((self.0 * other.0) / 100)
    }
}

impl QQuantity {
    /// Convert Quantity to QQuantity
    pub fn from_quantity(quantity: Quantity) -> Self {
        let quantized = quantity.round().to_i16().unwrap_or(0);
        QQuantity(quantized)
    }

    /// Convert QQuantity back to Quantity
    pub fn to_quantity(self) -> Quantity {
        Decimal::from(self.0)
    }
}

/// Order types
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderType {
    Market = 0,
    Limit = 1,
    Stop = 2,
    StopLimit = 3,
    TrailingStop = 4,
}

/// Order side
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

/// Order status
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderStatus {
    Pending = 0,
    PartialFill = 1,
    Filled = 2,
    Cancelled = 3,
    Rejected = 4,
    Expired = 5,
}

/// Time in force
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TimeInForce {
    Day = 0,
    Gtc = 1,        // Good Till Cancelled
    Ioc = 2,        // Immediate Or Cancel
    Fok = 3,        // Fill Or Kill
    Gtd = 4,        // Good Till Date
}

/// High-performance order structure
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Order {
    pub order_id: OrderId,
    pub symbol_id: SymbolId,
    pub order_type: OrderType,
    pub side: OrderSide,
    pub price: Price,
    pub quantity: Quantity,
    pub filled_quantity: Quantity,
    pub status: OrderStatus,
    pub create_time: Timestamp,
    pub update_time: Timestamp,
    pub venue_id: VenueId,
    pub account_id: u32,
    pub time_in_force: TimeInForce,
    pub flags: u16,

    // Computed fields for performance
    pub avg_fill_price: Price,
    pub leaves_quantity: Quantity,
}

impl Default for Order {
    fn default() -> Self {
        Self {
            order_id: 0,
            symbol_id: 0,
            order_type: OrderType::Market,
            side: OrderSide::Buy,
            price: Decimal::ZERO,
            quantity: Decimal::ZERO,
            filled_quantity: Decimal::ZERO,
            status: OrderStatus::Pending,
            create_time: current_timestamp(),
            update_time: current_timestamp(),
            venue_id: 0,
            account_id: 0,
            time_in_force: TimeInForce::Day,
            flags: 0,
            avg_fill_price: Decimal::ZERO,
            leaves_quantity: Decimal::ZERO,
        }
    }
}

impl Order {
    /// Update order with a fill
    pub fn update_fill(&mut self, fill_price: Price, fill_quantity: Quantity) {
        self.filled_quantity += fill_quantity;
        self.leaves_quantity = self.quantity - self.filled_quantity;

        // Update average fill price
        if self.filled_quantity > Decimal::ZERO {
            self.avg_fill_price = (self.avg_fill_price * (self.filled_quantity - fill_quantity) +
                                 fill_price * fill_quantity) / self.filled_quantity;
        } else {
            self.avg_fill_price = fill_price;
        }

        self.update_time = current_timestamp();

        if self.leaves_quantity == Decimal::ZERO {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartialFill;
        }
    }
}

/// Signal types
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(i8)]
pub enum SignalType {
    Long = 1,
    Short = -1,
    Neutral = 0,
    ExitLong = -2,
    ExitShort = 2,
}

/// High-performance signal structure
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Signal {
    pub symbol_id: SymbolId,
    pub signal: SignalType,
    pub confidence: f32,  // 0.0 to 1.0
    pub timestamp: Timestamp,
    pub strategy_id: u32,
    pub target_price: Price,
    pub target_quantity: Quantity,
}

impl Default for Signal {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            signal: SignalType::Neutral,
            confidence: 0.0,
            timestamp: current_timestamp(),
            strategy_id: 0,
            target_price: Decimal::ZERO,
            target_quantity: Decimal::ZERO,
        }
    }
}

/// Risk metrics structure
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct RiskMetrics {
    pub symbol_id: SymbolId,
    pub current_price: Price,
    pub position_size: Quantity,
    pub entry_price: Price,
    pub stop_loss_price: Price,
    pub take_profit_price: Price,
    pub var_95: f64,  // Value at Risk 95%
    pub expected_shortfall: f64,  // CVaR
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub timestamp: Timestamp,

    // Computed risk values
    pub unrealized_pnl: Price,
    pub position_var: f64,
    pub breach_warning: bool,
}

impl RiskMetrics {
    /// Update position metrics
    pub fn update_position(&mut self, current_px: Price) {
        self.current_price = current_px;
        self.unrealized_pnl = (current_px - self.entry_price) * self.position_size;
        self.position_var = self.var_95 * self.position_size.to_f64().unwrap_or(0.0);
        self.timestamp = current_timestamp();
    }

    /// Check stop loss condition
    pub fn check_stop_loss(&self) -> bool {
        if self.position_size > Decimal::ZERO {
            // Long position
            self.current_price <= self.stop_loss_price
        } else {
            // Short position
            self.current_price >= self.stop_loss_price
        }
    }

    /// Check take profit condition
    pub fn check_take_profit(&self) -> bool {
        if self.position_size > Decimal::ZERO {
            // Long position
            self.current_price >= self.take_profit_price
        } else {
            // Short position
            self.current_price <= self.take_profit_price
        }
    }
}

/// Performance counters for monitoring
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct PerformanceCounters {
    pub signals_processed: u64,
    pub orders_submitted: u64,
    pub orders_executed: u64,
    pub risk_checks_performed: u64,
    pub market_data_processed: u64,

    // Latency measurements (in nanoseconds)
    pub avg_signal_latency: u64,
    pub avg_order_latency: u64,
    pub avg_risk_latency: u64,
    pub max_signal_latency: u64,
    pub max_order_latency: u64,
    pub max_risk_latency: u64,

    // Error counters
    pub signal_errors: u64,
    pub order_errors: u64,
    pub risk_errors: u64,
    pub data_errors: u64,
}

impl Default for PerformanceCounters {
    fn default() -> Self {
        Self {
            signals_processed: 0,
            orders_submitted: 0,
            orders_executed: 0,
            risk_checks_performed: 0,
            market_data_processed: 0,
            avg_signal_latency: 0,
            avg_order_latency: 0,
            avg_risk_latency: 0,
            max_signal_latency: 0,
            max_order_latency: 0,
            max_risk_latency: 0,
            signal_errors: 0,
            order_errors: 0,
            risk_errors: 0,
            data_errors: 0,
        }
    }
}

impl PerformanceCounters {
    /// Update signal latency with exponential moving average
    pub fn update_signal_latency(&mut self, latency_ns: u64) {
        self.avg_signal_latency = (self.avg_signal_latency * 99 + latency_ns) / 100;
        if latency_ns > self.max_signal_latency {
            self.max_signal_latency = latency_ns;
        }
    }

    /// Update order latency with exponential moving average
    pub fn update_order_latency(&mut self, latency_ns: u64) {
        self.avg_order_latency = (self.avg_order_latency * 99 + latency_ns) / 100;
        if latency_ns > self.max_order_latency {
            self.max_order_latency = latency_ns;
        }
    }

    /// Update risk latency with exponential moving average
    pub fn update_risk_latency(&mut self, latency_ns: u64) {
        self.avg_risk_latency = (self.avg_risk_latency * 99 + latency_ns) / 100;
        if latency_ns > self.max_risk_latency {
            self.max_risk_latency = latency_ns;
        }
    }
}

/// Constants
pub const MAX_SYMBOLS: usize = 10_000;
pub const MAX_ORDERS: usize = 1_000_000;
pub const MAX_SIGNALS: usize = 100_000;
pub const CACHE_LINE_SIZE: usize = 64;

/// Utility functions
pub fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as Timestamp
}

pub fn price_from_float(price: f64) -> Price {
    Decimal::from_f64(price).unwrap_or(Decimal::ZERO)
}

pub fn price_to_float(price: Price) -> f64 {
    price.to_f64().unwrap_or(0.0)
}

/// Error types
#[derive(Debug, thiserror::Error)]
pub enum TradingError {
    #[error("Invalid order: {message}")]
    InvalidOrder { message: String },

    #[error("Risk limit breached: {limit}")]
    RiskLimitBreached { limit: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Data error: {message}")]
    DataError { message: String },

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

pub type Result<T> = std::result::Result<T, TradingError>;
