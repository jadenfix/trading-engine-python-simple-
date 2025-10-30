package core

import (
	"go-trading-engine/pkg/types"
	"go-trading-engine/pkg/quantization"
	"go-trading-engine/pkg/memory"
	"go-trading-engine/pkg/cache"
)

// Re-export types for convenience
type (
	MarketData = types.MarketData
	Order = types.Order
	Signal = types.Signal
	RiskMetrics = types.RiskMetrics
	PerformanceCounters = types.PerformanceCounters
	TradingResult = types.TradingResult
	RiskCheckResult = types.RiskCheckResult
	TradingError = types.TradingError
	Price = types.Price
	Quantity = types.Quantity
	Timestamp = types.Timestamp
	OrderID = types.OrderID
	SymbolID = types.SymbolID
	VenueID = types.VenueID
	OrderType = types.OrderType
	OrderSide = types.OrderSide
	OrderStatus = types.OrderStatus
	TimeInForce = types.TimeInForce
	SignalType = types.SignalType
	QPrice = types.QPrice
	QQuantity = types.QQuantity
)

// Re-export utility functions
var (
	CurrentTimestamp = types.CurrentTimestamp
	DecimalFromFloat = types.DecimalFromFloat
	PriceFromFloat = types.PriceFromFloat
	QuantityFromFloat = types.QuantityFromFloat
	NewTradingError = types.NewTradingError
)

// Trading engine configuration
type TradingEngineConfig struct {
	MaxSignalsPerSymbol     int     `json:"max_signals_per_symbol"`
	MinSignalConfidence     float32 `json:"min_signal_confidence"`
	MaxSignalAgeNS          int64   `json:"max_signal_age_ns"`
	EnableSIMD              bool    `json:"enable_simd"`
	EnableQuantization      bool    `json:"enable_quantization"`
	EnableCaching           bool    `json:"enable_caching"`
	MaxPortfolioVaR         float64 `json:"max_portfolio_var"`
	MaxPositionVaR          float64 `json:"max_position_var"`
	MaxDrawdown             float64 `json:"max_drawdown"`
	MaxLeverage             float64 `json:"max_leverage"`
	MaxConcentration        float64 `json:"max_concentration"`
	MaxPositions            int     `json:"max_positions"`
}

func DefaultConfig() *TradingEngineConfig {
	return &TradingEngineConfig{
		MaxSignalsPerSymbol: 10,
		MinSignalConfidence: 0.1,
		MaxSignalAgeNS:      1000000000, // 1 second
		EnableSIMD:          true,
		EnableQuantization:  true,
		EnableCaching:       true,
		MaxPortfolioVaR:     0.05, // 5% VaR limit
		MaxPositionVaR:      0.02, // 2% per position VaR
		MaxDrawdown:         0.10, // 10% max drawdown
		MaxLeverage:         5.0,  // 5x max leverage
		MaxConcentration:    0.25, // 25% max position size
		MaxPositions:        100,  // Max number of positions
	}
}

// Global instances for performance
var (
	GlobalQuantizer = quantization.NewPriceQuantizer(100) // 2 decimal places
	GlobalCache     = cache.NewMultiLevelCache()
	GlobalPool      = memory.NewMemoryPool(64, 1000) // 64-byte blocks, 1000 blocks
)

// Initialize global instances
func init() {
	// Pre-warm caches and pools
	GlobalCache.PreloadSymbol(1, []Price{PriceFromFloat(100.0), PriceFromFloat(101.0)})
}

// Utility functions for high-performance operations
func FastPriceComparison(a, b Price) int {
	return a.Cmp(b)
}

func FastQuantityAdd(a, b Quantity) Quantity {
	return a.Add(b)
}

func FastPriceMultiply(price Price, factor float64) Price {
	return price.Mul(DecimalFromFloat(factor))
}

// Memory-efficient slice operations
func FilterSignals(signals []Signal, predicate func(Signal) bool) []Signal {
	result := make([]Signal, 0, len(signals))
	for _, signal := range signals {
		if predicate(signal) {
			result = append(result, signal)
		}
	}
	return result
}

func GroupSignalsBySymbol(signals []Signal) map[SymbolID][]Signal {
	groups := make(map[SymbolID][]Signal)
	for _, signal := range signals {
		groups[signal.SymbolID] = append(groups[signal.SymbolID], signal)
	}
	return groups
}

// Risk calculation utilities
func CalculateUnrealizedPnL(positionSize Quantity, entryPrice, currentPrice Price) Price {
	if positionSize.IsZero() {
		return PriceFromFloat(0)
	}
	return positionSize.Mul(currentPrice.Sub(entryPrice))
}

func CalculatePositionVaR(positionSize Quantity, var95 float64) float64 {
	positionSizeFloat, _ := positionSize.Float64()
	return var95 * positionSizeFloat
}

// Validation functions
func ValidateOrder(order *Order) error {
	if order.SymbolID == 0 {
		return NewTradingError("INVALID_ORDER", "SymbolID cannot be zero")
	}
	if order.Quantity.IsZero() || order.Quantity.IsNegative() {
		return NewTradingError("INVALID_ORDER", "Quantity must be positive")
	}
	if order.OrderType == types.OrderTypeLimit && order.Price.IsZero() {
		return NewTradingError("INVALID_ORDER", "Limit orders must have a price")
	}
	return nil
}

func ValidateSignal(signal *Signal) error {
	if signal.SymbolID == 0 {
		return NewTradingError("INVALID_SIGNAL", "SymbolID cannot be zero")
	}
	if signal.Confidence < 0 || signal.Confidence > 1 {
		return NewTradingError("INVALID_SIGNAL", "Confidence must be between 0 and 1")
	}
	if signal.TargetQuantity.IsZero() {
		return NewTradingError("INVALID_SIGNAL", "Target quantity cannot be zero")
	}
	return nil
}

// Performance optimization hints for Go compiler
//go:noinline
func HeavyComputationHint() {
	// This function is marked noinline to prevent inlining
	// of heavy computations, allowing better profiling
}
