package performance

import (
	"sync/atomic"
	"time"
)

// PerformanceCounters tracks performance metrics
type PerformanceCounters struct {
	SignalsProcessed         int64 `json:"signals_processed"`
	OrdersSubmitted          int64 `json:"orders_submitted"`
	OrdersExecuted           int64 `json:"orders_executed"`
	RiskChecksPerformed      int64 `json:"risk_checks_performed"`
	MarketDataProcessed      int64 `json:"market_data_processed"`

	// Latency measurements (in nanoseconds)
	AvgSignalLatencyNS       int64 `json:"avg_signal_latency_ns"`
	AvgOrderLatencyNS        int64 `json:"avg_order_latency_ns"`
	AvgRiskLatencyNS         int64 `json:"avg_risk_latency_ns"`
	MaxSignalLatencyNS       int64 `json:"max_signal_latency_ns"`
	MaxOrderLatencyNS        int64 `json:"max_order_latency_ns"`
	MaxRiskLatencyNS         int64 `json:"max_risk_latency_ns"`

	// Error counters
	SignalErrors             int64 `json:"signal_errors"`
	OrderErrors              int64 `json:"order_errors"`
	RiskErrors               int64 `json:"risk_errors"`
	DataErrors               int64 `json:"data_errors"`
}

// PerformanceMonitor provides performance monitoring capabilities
type PerformanceMonitor struct {
	counters *PerformanceCounters
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		counters: &PerformanceCounters{},
	}
}

// GetCounters returns the performance counters
func (pm *PerformanceMonitor) GetCounters() *PerformanceCounters {
	return pm.counters
}

// IncrementSignalsProcessed increments signal processing counter
func (pm *PerformanceMonitor) IncrementSignalsProcessed(count int) {
	atomic.AddInt64(&pm.counters.SignalsProcessed, int64(count))
}

// AddSignalsProcessed adds to signals processed counter
func (pm *PerformanceMonitor) AddSignalsProcessed(count int64) {
	atomic.AddInt64(&pm.counters.SignalsProcessed, count)
}

// IncrementOrdersSubmitted increments orders submitted counter
func (pm *PerformanceMonitor) IncrementOrdersSubmitted() {
	atomic.AddInt64(&pm.counters.OrdersSubmitted, 1)
}

// AddOrdersSubmitted adds to orders submitted counter
func (pm *PerformanceMonitor) AddOrdersSubmitted(count int64) {
	atomic.AddInt64(&pm.counters.OrdersSubmitted, count)
}

// IncrementOrdersExecuted increments orders executed counter
func (pm *PerformanceMonitor) IncrementOrdersExecuted() {
	atomic.AddInt64(&pm.counters.OrdersExecuted, 1)
}

// AddOrdersExecuted adds to orders executed counter
func (pm *PerformanceMonitor) AddOrdersExecuted(count int64) {
	atomic.AddInt64(&pm.counters.OrdersExecuted, count)
}

// IncrementRiskChecksPerformed increments risk checks counter
func (pm *PerformanceMonitor) IncrementRiskChecksPerformed() {
	atomic.AddInt64(&pm.counters.RiskChecksPerformed, 1)
}

// AddRiskChecksPerformed adds to risk checks counter
func (pm *PerformanceMonitor) AddRiskChecksPerformed(count int64) {
	atomic.AddInt64(&pm.counters.RiskChecksPerformed, count)
}

// IncrementMarketDataProcessed increments market data processed counter
func (pm *PerformanceMonitor) IncrementMarketDataProcessed(count int) {
	atomic.AddInt64(&pm.counters.MarketDataProcessed, int64(count))
}

// AddMarketDataProcessed adds to market data processed counter
func (pm *PerformanceMonitor) AddMarketDataProcessed(count int64) {
	atomic.AddInt64(&pm.counters.MarketDataProcessed, count)
}

// UpdateSignalLatency updates signal processing latency
func (pm *PerformanceMonitor) UpdateSignalLatency(latency time.Duration) {
	latencyNS := latency.Nanoseconds()

	// Update max latency
	for {
		currentMax := atomic.LoadInt64(&pm.counters.MaxSignalLatencyNS)
		if latencyNS <= currentMax || atomic.CompareAndSwapInt64(&pm.counters.MaxSignalLatencyNS, currentMax, latencyNS) {
			break
		}
	}

	// Update average latency (exponential moving average)
	for {
		currentAvg := atomic.LoadInt64(&pm.counters.AvgSignalLatencyNS)
		newAvg := (currentAvg*99 + latencyNS) / 100
		if atomic.CompareAndSwapInt64(&pm.counters.AvgSignalLatencyNS, currentAvg, newAvg) {
			break
		}
	}
}

// UpdateOrderLatency updates order processing latency
func (pm *PerformanceMonitor) UpdateOrderLatency(latency time.Duration) {
	latencyNS := latency.Nanoseconds()

	// Update max latency
	for {
		currentMax := atomic.LoadInt64(&pm.counters.MaxOrderLatencyNS)
		if latencyNS <= currentMax || atomic.CompareAndSwapInt64(&pm.counters.MaxOrderLatencyNS, currentMax, latencyNS) {
			break
		}
	}

	// Update average latency
	for {
		currentAvg := atomic.LoadInt64(&pm.counters.AvgOrderLatencyNS)
		newAvg := (currentAvg*99 + latencyNS) / 100
		if atomic.CompareAndSwapInt64(&pm.counters.AvgOrderLatencyNS, currentAvg, newAvg) {
			break
		}
	}
}

// UpdateRiskLatency updates risk calculation latency
func (pm *PerformanceMonitor) UpdateRiskLatency(latency time.Duration) {
	latencyNS := latency.Nanoseconds()

	// Update max latency
	for {
		currentMax := atomic.LoadInt64(&pm.counters.MaxRiskLatencyNS)
		if latencyNS <= currentMax || atomic.CompareAndSwapInt64(&pm.counters.MaxRiskLatencyNS, currentMax, latencyNS) {
			break
		}
	}

	// Update average latency
	for {
		currentAvg := atomic.LoadInt64(&pm.counters.AvgRiskLatencyNS)
		newAvg := (currentAvg*99 + latencyNS) / 100
		if atomic.CompareAndSwapInt64(&pm.counters.AvgRiskLatencyNS, currentAvg, newAvg) {
			break
		}
	}
}

// IncrementSignalErrors increments signal error counter
func (pm *PerformanceMonitor) IncrementSignalErrors() {
	atomic.AddInt64(&pm.counters.SignalErrors, 1)
}

// IncrementOrderErrors increments order error counter
func (pm *PerformanceMonitor) IncrementOrderErrors() {
	atomic.AddInt64(&pm.counters.OrderErrors, 1)
}

// IncrementRiskErrors increments risk error counter
func (pm *PerformanceMonitor) IncrementRiskErrors() {
	atomic.AddInt64(&pm.counters.RiskErrors, 1)
}

// IncrementDataErrors increments data error counter
func (pm *PerformanceMonitor) IncrementDataErrors() {
	atomic.AddInt64(&pm.counters.DataErrors, 1)
}

// Reset resets all performance counters
func (pm *PerformanceMonitor) Reset() {
	atomic.StoreInt64(&pm.counters.SignalsProcessed, 0)
	atomic.StoreInt64(&pm.counters.OrdersSubmitted, 0)
	atomic.StoreInt64(&pm.counters.OrdersExecuted, 0)
	atomic.StoreInt64(&pm.counters.RiskChecksPerformed, 0)
	atomic.StoreInt64(&pm.counters.MarketDataProcessed, 0)
	atomic.StoreInt64(&pm.counters.AvgSignalLatencyNS, 0)
	atomic.StoreInt64(&pm.counters.AvgOrderLatencyNS, 0)
	atomic.StoreInt64(&pm.counters.AvgRiskLatencyNS, 0)
	atomic.StoreInt64(&pm.counters.MaxSignalLatencyNS, 0)
	atomic.StoreInt64(&pm.counters.MaxOrderLatencyNS, 0)
	atomic.StoreInt64(&pm.counters.MaxRiskLatencyNS, 0)
	atomic.StoreInt64(&pm.counters.SignalErrors, 0)
	atomic.StoreInt64(&pm.counters.OrderErrors, 0)
	atomic.StoreInt64(&pm.counters.RiskErrors, 0)
	atomic.StoreInt64(&pm.counters.DataErrors, 0)
}

// GetStats returns performance statistics as a map
func (pm *PerformanceMonitor) GetStats() map[string]interface{} {
	counters := pm.counters

	stats := make(map[string]interface{})

	// Throughput metrics
	stats["signals_processed"] = atomic.LoadInt64(&counters.SignalsProcessed)
	stats["orders_submitted"] = atomic.LoadInt64(&counters.OrdersSubmitted)
	stats["orders_executed"] = atomic.LoadInt64(&counters.OrdersExecuted)
	stats["risk_checks_performed"] = atomic.LoadInt64(&counters.RiskChecksPerformed)
	stats["market_data_processed"] = atomic.LoadInt64(&counters.MarketDataProcessed)

	// Latency metrics (convert to microseconds for readability)
	stats["avg_signal_latency_us"] = float64(atomic.LoadInt64(&counters.AvgSignalLatencyNS)) / 1000.0
	stats["avg_order_latency_us"] = float64(atomic.LoadInt64(&counters.AvgOrderLatencyNS)) / 1000.0
	stats["avg_risk_latency_us"] = float64(atomic.LoadInt64(&counters.AvgRiskLatencyNS)) / 1000.0
	stats["max_signal_latency_us"] = float64(atomic.LoadInt64(&counters.MaxSignalLatencyNS)) / 1000.0
	stats["max_order_latency_us"] = float64(atomic.LoadInt64(&counters.MaxOrderLatencyNS)) / 1000.0
	stats["max_risk_latency_us"] = float64(atomic.LoadInt64(&counters.MaxRiskLatencyNS)) / 1000.0

	// Error metrics
	stats["signal_errors"] = atomic.LoadInt64(&counters.SignalErrors)
	stats["order_errors"] = atomic.LoadInt64(&counters.OrderErrors)
	stats["risk_errors"] = atomic.LoadInt64(&counters.RiskErrors)
	stats["data_errors"] = atomic.LoadInt64(&counters.DataErrors)

	// Calculated metrics
	signalsProcessed := atomic.LoadInt64(&counters.SignalsProcessed)
	ordersProcessed := atomic.LoadInt64(&counters.OrdersSubmitted)

	if signalsProcessed > 0 {
		stats["error_rate_signals"] = float64(atomic.LoadInt64(&counters.SignalErrors)) / float64(signalsProcessed)
	}

	if ordersProcessed > 0 {
		stats["error_rate_orders"] = float64(atomic.LoadInt64(&counters.OrderErrors)) / float64(ordersProcessed)
	}

	return stats
}
