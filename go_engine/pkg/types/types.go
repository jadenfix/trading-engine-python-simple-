package types

import (
	"time"
	"sync/atomic"

	"github.com/shopspring/decimal"
)

// Financial types with high precision
type Price = decimal.Decimal      // High-precision price representation
type Quantity = decimal.Decimal   // High-precision quantity
type Timestamp = int64            // Nanoseconds since epoch
type OrderID = uint64             // Unique order identifier
type SymbolID = uint32            // Internal symbol identifier
type VenueID = uint16             // Trading venue identifier

// Quantized types for ultra-fast computation
type QPrice int32   // Quantized price (precision: 0.01)
type QQuantity int16 // Quantized quantity (precision: 1)

// High-performance market data structure
type MarketData struct {
	SymbolID   SymbolID `json:"symbol_id"`
	BidPrice   Price    `json:"bid_price"`
	AskPrice   Price    `json:"ask_price"`
	BidSize    Quantity `json:"bid_size"`
	AskSize    Quantity `json:"ask_size"`
	Timestamp  Timestamp `json:"timestamp"`
	VenueID    VenueID  `json:"venue_id"`
	Flags      uint16   `json:"flags"`
}

// Quantized market data for SIMD processing
type MarketDataQuantized struct {
	SymbolID   SymbolID `json:"symbol_id"`
	BidPrice   QPrice   `json:"bid_price"`
	AskPrice   QPrice   `json:"ask_price"`
	BidSize    QQuantity `json:"bid_size"`
	AskSize    QQuantity `json:"ask_size"`
	Timestamp  Timestamp `json:"timestamp"`
	VenueID    VenueID  `json:"venue_id"`
	Flags      uint16   `json:"flags"`
}

// Convert methods for quantized types
func (q QPrice) ToPrice() Price {
	return decimal.NewFromInt32(int32(q)).Div(decimal.NewFromInt(100))
}

func PriceToQPrice(price Price) QPrice {
	scaled := price.Mul(decimal.NewFromInt(100))
	intVal, _ := scaled.Int64()
	return QPrice(int32(intVal))
}

func (q QQuantity) ToQuantity() Quantity {
	return decimal.NewFromInt16(int16(q))
}

func QuantityToQQuantity(quantity Quantity) QQuantity {
	intVal, _ := quantity.Int64()
	return QQuantity(int16(intVal))
}

// Order types
type OrderType uint8

const (
	OrderTypeMarket OrderType = iota
	OrderTypeLimit
	OrderTypeStop
	OrderTypeStopLimit
	OrderTypeTrailingStop
)

// Order side
type OrderSide uint8

const (
	OrderSideBuy OrderSide = iota
	OrderSideSell
)

// Order status
type OrderStatus uint8

const (
	OrderStatusPending OrderStatus = iota
	OrderStatusPartialFill
	OrderStatusFilled
	OrderStatusCancelled
	OrderStatusRejected
	OrderStatusExpired
)

// Time in force
type TimeInForce uint8

const (
	TimeInForceDay TimeInForce = iota
	TimeInForceGTC
	TimeInForceIOC
	TimeInForceFOK
	TimeInForceGTD
)

// High-performance order structure
type Order struct {
	OrderID      OrderID     `json:"order_id"`
	SymbolID     SymbolID    `json:"symbol_id"`
	OrderType    OrderType   `json:"order_type"`
	Side         OrderSide   `json:"side"`
	Price        Price       `json:"price"`
	Quantity     Quantity    `json:"quantity"`
	FilledQuantity Quantity  `json:"filled_quantity"`
	Status       OrderStatus `json:"status"`
	CreateTime   Timestamp   `json:"create_time"`
	UpdateTime   Timestamp   `json:"update_time"`
	VenueID      VenueID     `json:"venue_id"`
	AccountID    uint32      `json:"account_id"`
	TimeInForce  TimeInForce `json:"time_in_force"`
	Flags        uint16      `json:"flags"`

	// Computed fields for performance
	AvgFillPrice Price    `json:"avg_fill_price"`
	LeavesQuantity Quantity `json:"leaves_quantity"`
}

// Signal types
type SignalType int8

const (
	SignalNeutral SignalType = iota - 1
	SignalShort
	SignalExitLong
	SignalLong
	SignalExitShort
)

// High-performance signal structure
type Signal struct {
	SymbolID        SymbolID   `json:"symbol_id"`
	Signal          SignalType `json:"signal"`
	Confidence      float32    `json:"confidence"`
	Timestamp       Timestamp  `json:"timestamp"`
	StrategyID      uint32     `json:"strategy_id"`
	TargetPrice     Price      `json:"target_price"`
	TargetQuantity  Quantity   `json:"target_quantity"`
}

// Risk metrics structure
type RiskMetrics struct {
	SymbolID           SymbolID  `json:"symbol_id"`
	CurrentPrice       Price     `json:"current_price"`
	PositionSize       Quantity  `json:"position_size"`
	EntryPrice         Price     `json:"entry_price"`
	StopLossPrice      Price     `json:"stop_loss_price"`
	TakeProfitPrice    Price     `json:"take_profit_price"`
	VaR95              float64   `json:"var_95"`
	ExpectedShortfall  float64   `json:"expected_shortfall"`
	SharpeRatio        float64   `json:"sharpe_ratio"`
	MaxDrawdown        float64   `json:"max_drawdown"`
	Timestamp          Timestamp `json:"timestamp"`

	// Computed risk values
	UnrealizedPnL      Price     `json:"unrealized_pnl"`
	PositionVaR        float64   `json:"position_var"`
	BreachWarning      bool      `json:"breach_warning"`
}

// Performance counters for monitoring
type PerformanceCounters struct {
	SignalsProcessed       int64 `json:"signals_processed"`
	OrdersSubmitted        int64 `json:"orders_submitted"`
	OrdersExecuted         int64 `json:"orders_executed"`
	RiskChecksPerformed    int64 `json:"risk_checks_performed"`
	MarketDataProcessed    int64 `json:"market_data_processed"`

	// Latency measurements (in nanoseconds)
	AvgSignalLatencyNS     int64 `json:"avg_signal_latency_ns"`
	AvgOrderLatencyNS      int64 `json:"avg_order_latency_ns"`
	AvgRiskLatencyNS       int64 `json:"avg_risk_latency_ns"`
	MaxSignalLatencyNS     int64 `json:"max_signal_latency_ns"`
	MaxOrderLatencyNS      int64 `json:"max_order_latency_ns"`
	MaxRiskLatencyNS       int64 `json:"max_risk_latency_ns"`

	// Error counters
	SignalErrors           int64 `json:"signal_errors"`
	OrderErrors            int64 `json:"order_errors"`
	RiskErrors             int64 `json:"risk_errors"`
	DataErrors             int64 `json:"data_errors"`
}

// Trading result
type TradingResult struct {
	Signals              []Signal `json:"signals"`
	Orders               []Order  `json:"orders"`
	ProcessingTimeNS     int64    `json:"processing_time_ns"`
	MarketDataProcessed  int      `json:"market_data_processed"`
}

// Risk check result
type RiskCheckResult struct {
	IsValid    bool     `json:"is_valid"`
	Violations []string `json:"violations,omitempty"`
}

// Constants
const (
	MaxSymbols   = 10000
	MaxOrders    = 1000000
	MaxSignals   = 100000
	CacheLineSize = 64
)

// Utility functions
var timestampOffset int64

func init() {
	// Initialize timestamp offset to Unix epoch
	timestampOffset = time.Now().UnixNano() - time.Now().UnixNano()
}

func CurrentTimestamp() Timestamp {
	return Timestamp(time.Now().UnixNano() + atomic.LoadInt64(&timestampOffset))
}

func DecimalFromFloat(f float64) decimal.Decimal {
	return decimal.NewFromFloat(f)
}

func PriceFromFloat(f float64) Price {
	return decimal.NewFromFloat(f)
}

func QuantityFromFloat(f float64) Quantity {
	return decimal.NewFromFloat(f)
}

// Error types
type TradingError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e TradingError) Error() string {
	return e.Message
}

func NewTradingError(code, message string) *TradingError {
	return &TradingError{Code: code, Message: message}
}
