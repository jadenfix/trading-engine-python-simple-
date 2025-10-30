package data_engine

import (
	"go-trading-engine/internal/core"
	"sync"
	"time"
)

// DataEngine handles market data processing and storage
type DataEngine struct {
	dataCache    map[core.SymbolID][]core.MarketData
	maxHistory   int
	mutex        sync.RWMutex
	performanceCounters *core.PerformanceCounters
}

// NewDataEngine creates a new data engine
func NewDataEngine() *DataEngine {
	return &DataEngine{
		dataCache:    make(map[core.SymbolID][]core.MarketData),
		maxHistory:   1000,
		performanceCounters: &core.PerformanceCounters{},
	}
}

// ProcessMarketData processes incoming market data
func (de *DataEngine) ProcessMarketData(marketData []core.MarketData) []core.MarketData {
	start := time.Now()

	de.mutex.Lock()
	defer de.mutex.Unlock()

	processed := make([]core.MarketData, len(marketData))

	for i, data := range marketData {
		// Normalize data (add any preprocessing here)
		normalized := de.normalizeMarketData(data)

		// Store in cache
		history := de.dataCache[data.SymbolID]
		history = append(history, normalized)

		// Keep only recent history
		if len(history) > de.maxHistory {
			// Remove oldest entries
			history = history[len(history)-de.maxHistory:]
		}

		de.dataCache[data.SymbolID] = history
		processed[i] = normalized
	}

	processingTime := time.Since(start)
	de.performanceCounters.UpdateDataProcessingLatency(processingTime)
	de.performanceCounters.MarketDataProcessed += int64(len(marketData))

	return processed
}

// normalizeMarketData normalizes market data
func (de *DataEngine) normalizeMarketData(data core.MarketData) core.MarketData {
	// Apply any data normalization/cleaning here
	normalized := data

	// Ensure prices are positive
	if normalized.BidPrice.IsNegative() {
		normalized.BidPrice = core.PriceFromFloat(0)
	}
	if normalized.AskPrice.IsNegative() {
		normalized.AskPrice = core.PriceFromFloat(0)
	}

	// Ensure bid <= ask
	if normalized.BidPrice.GreaterThan(normalized.AskPrice) {
		// Swap them if bid > ask
		normalized.BidPrice, normalized.AskPrice = normalized.AskPrice, normalized.BidPrice
	}

	// Ensure volumes are positive
	if normalized.BidSize.IsNegative() {
		normalized.BidSize = core.QuantityFromFloat(0)
	}
	if normalized.AskSize.IsNegative() {
		normalized.AskSize = core.QuantityFromFloat(0)
	}

	return normalized
}

// GetHistoricalData retrieves historical data for a symbol
func (de *DataEngine) GetHistoricalData(symbolID core.SymbolID, maxPoints int) []core.MarketData {
	de.mutex.RLock()
	defer de.mutex.RUnlock()

	if history, exists := de.dataCache[symbolID]; exists {
		start := len(history) - maxPoints
		if start < 0 {
			start = 0
		}

		result := make([]core.MarketData, len(history)-start)
		copy(result, history[start:])
		return result
	}

	return nil
}

// GetLatestData retrieves the latest market data for a symbol
func (de *DataEngine) GetLatestData(symbolID core.SymbolID) (*core.MarketData, bool) {
	de.mutex.RLock()
	defer de.mutex.RUnlock()

	if history, exists := de.dataCache[symbolID]; exists && len(history) > 0 {
		latest := history[len(history)-1]
		return &latest, true
	}

	return nil, false
}

// GetPriceHistory returns price history for technical analysis
func (de *DataEngine) GetPriceHistory(symbolID core.SymbolID, maxPoints int) []core.Price {
	de.mutex.RLock()
	defer de.mutex.RUnlock()

	if history, exists := de.dataCache[symbolID]; exists {
		start := len(history) - maxPoints
		if start < 0 {
			start = 0
		}

		prices := make([]core.Price, 0, len(history)-start)
		for i := start; i < len(history); i++ {
			prices = append(prices, history[i].AskPrice) // Use ask price as reference
		}

		return prices
	}

	return nil
}

// GetVolumeHistory returns volume history for analysis
func (de *DataEngine) GetVolumeHistory(symbolID core.SymbolID, maxPoints int) []core.Quantity {
	de.mutex.RLock()
	defer de.mutex.RUnlock()

	if history, exists := de.dataCache[symbolID]; exists {
		start := len(history) - maxPoints
		if start < 0 {
			start = 0
		}

		volumes := make([]core.Quantity, 0, len(history)-start)
		for i := start; i < len(history); i++ {
			volumes = append(volumes, history[i].AskSize) // Use ask size as reference
		}

		return volumes
	}

	return nil
}

// GetSymbols returns all tracked symbols
func (de *DataEngine) GetSymbols() []core.SymbolID {
	de.mutex.RLock()
	defer de.mutex.RUnlock()

	symbols := make([]core.SymbolID, 0, len(de.dataCache))
	for symbolID := range de.dataCache {
		symbols = append(symbols, symbolID)
	}

	return symbols
}

// GetDataStats returns data statistics
func (de *DataEngine) GetDataStats() map[string]interface{} {
	de.mutex.RLock()
	defer de.mutex.RUnlock()

	stats := make(map[string]interface{})
	stats["total_symbols"] = len(de.dataCache)
	stats["max_history"] = de.maxHistory

	totalPoints := 0
	for _, history := range de.dataCache {
		totalPoints += len(history)
	}
	stats["total_data_points"] = totalPoints

	return stats
}

// ClearData clears all stored data
func (de *DataEngine) ClearData() {
	de.mutex.Lock()
	defer de.mutex.Unlock()

	de.dataCache = make(map[core.SymbolID][]core.MarketData)
}

// ClearSymbolData clears data for a specific symbol
func (de *DataEngine) ClearSymbolData(symbolID core.SymbolID) {
	de.mutex.Lock()
	defer de.mutex.Unlock()

	delete(de.dataCache, symbolID)
}

// SetMaxHistory sets the maximum history to keep per symbol
func (de *DataEngine) SetMaxHistory(maxHistory int) {
	de.mutex.Lock()
	defer de.mutex.Unlock()

	de.maxHistory = maxHistory

	// Trim existing data to new limit
	for symbolID, history := range de.dataCache {
		if len(history) > maxHistory {
			de.dataCache[symbolID] = history[len(history)-maxHistory:]
		}
	}
}

// GetPerformanceCounters returns performance statistics
func (de *DataEngine) GetPerformanceCounters() *core.PerformanceCounters {
	return de.performanceCounters
}

// ResetPerformanceCounters resets performance counters
func (de *DataEngine) ResetPerformanceCounters() {
	de.performanceCounters = &core.PerformanceCounters{}
}

// UpdateDataProcessingLatency updates data processing latency (extension to PerformanceCounters)
func (pc *core.PerformanceCounters) UpdateDataProcessingLatency(latency time.Duration) {
	// This would need to be added to the core PerformanceCounters struct
	// For now, we'll just update a generic latency field
	pc.UpdateSignalLatency(latency) // Reuse existing field
}
