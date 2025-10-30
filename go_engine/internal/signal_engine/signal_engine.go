package signal_engine

import (
	"go-trading-engine/internal/core"
	"go-trading-engine/pkg/cache"
	"go-trading-engine/pkg/types"
	"math"
	"time"
)

// SignalEngine provides high-performance signal generation
type SignalEngine struct {
	technicalIndicators *TechnicalIndicators
	config              *SignalConfig
	cache               *cache.MultiLevelCache
	performanceCounters *core.PerformanceCounters
}

// SignalConfig holds signal generation configuration
type SignalConfig struct {
	MaxSignalsPerSymbol   int     `json:"max_signals_per_symbol"`
	MinSignalConfidence   float32 `json:"min_signal_confidence"`
	MaxSignalAgeNS        int64   `json:"max_signal_age_ns"`
	EnableCaching         bool    `json:"enable_caching"`
	EnableQuantization    bool    `json:"enable_quantization"`
}

// NewSignalEngine creates a new signal engine
func NewSignalEngine() *SignalEngine {
	return &SignalEngine{
		technicalIndicators: NewTechnicalIndicators(),
		config: &SignalConfig{
			MaxSignalsPerSymbol: 10,
			MinSignalConfidence: 0.1,
			MaxSignalAgeNS:      1000000000, // 1 second
			EnableCaching:       true,
			EnableQuantization:  true,
		},
		cache:               cache.NewMultiLevelCache(),
		performanceCounters: &core.PerformanceCounters{},
	}
}

// GenerateSignals generates trading signals from market data
func (se *SignalEngine) GenerateSignals(marketData []core.MarketData) []core.Signal {
	start := time.Now()

	// Group market data by symbol
	symbolData := make(map[core.SymbolID][]core.MarketData)
	for _, data := range marketData {
		symbolData[data.SymbolID] = append(symbolData[data.SymbolID], data)
	}

	var allSignals []core.Signal

	// Generate signals for each symbol
	for symbolID, dataPoints := range symbolData {
		symbolSignals := se.generateSignalsForSymbol(symbolID, dataPoints)
		allSignals = append(allSignals, symbolSignals...)
	}

	// Apply post-processing and filtering
	filteredSignals := se.filterAndRankSignals(allSignals)

	// Update cache
	if se.config.EnableCaching {
		for _, signal := range filteredSignals {
			se.cache.SignalCache().CacheSignals(signal.SymbolID, signal.StrategyID, []core.Signal{signal})
		}
	}

	processingTime := time.Since(start)
	se.performanceCounters.UpdateSignalLatency(processingTime)
	se.performanceCounters.SignalsProcessed += int64(len(filteredSignals))

	return filteredSignals
}

// generateSignalsForSymbol generates signals for a specific symbol
func (se *SignalEngine) generateSignalsForSymbol(symbolID core.SymbolID, dataPoints []core.MarketData) []core.Signal {
	if len(dataPoints) < 20 {
		return nil // Need minimum data for signal generation
	}

	// Sort data by timestamp
	// (Assuming data comes pre-sorted, otherwise we'd sort here)

	// Extract price and volume series
	prices := make([]core.Price, len(dataPoints))
	volumes := make([]core.Quantity, len(dataPoints))

	for i, data := range dataPoints {
		prices[i] = data.AskPrice
		volumes[i] = data.AskSize
	}

	// Update price history cache
	if se.config.EnableCaching {
		for _, price := range prices {
			se.cache.PriceHistory().AddPrice(symbolID, price)
		}
	}

	var signals []core.Signal

	// Generate technical signals
	technicalSignals := se.generateTechnicalSignals(symbolID, prices, volumes)
	signals = append(signals, technicalSignals...)

	// Generate statistical signals
	statisticalSignals := se.generateStatisticalSignals(symbolID, prices)
	signals = append(signals, statisticalSignals...)

	// Generate ML-based signals
	mlSignals := se.generateMLSignals(symbolID, prices, volumes)
	signals = append(signals, mlSignals...)

	return signals
}

// generateTechnicalSignals generates technical indicator based signals
func (se *SignalEngine) generateTechnicalSignals(symbolID core.SymbolID, prices []core.Price, volumes []core.Quantity) []core.Signal {
	var signals []core.Signal

	if len(prices) < 20 {
		return signals
	}

	// Calculate indicators
	indicators := se.technicalIndicators.CalculateIndicators(prices, volumes)

	currentPrice := prices[len(prices)-1]
	timestamp := core.CurrentTimestamp()

	// SMA crossover signals
	if indicators.SMA20 > 0 && indicators.SMA50 > 0 {
		if indicators.SMA20 > indicators.SMA50 && indicators.SMA20 <= indicators.SMA50*1.02 {
			// Golden cross (SMA 20 crosses above SMA 50)
			signals = append(signals, core.Signal{
				SymbolID:   symbolID,
				Signal:     core.SignalLong,
				Confidence: 0.7,
				Timestamp:  timestamp,
				StrategyID: 1, // Technical strategy
				TargetPrice: currentPrice,
				TargetQuantity: core.QuantityFromFloat(100),
			})
		} else if indicators.SMA20 < indicators.SMA50 && indicators.SMA20 >= indicators.SMA50*0.98 {
			// Death cross (SMA 20 crosses below SMA 50)
			signals = append(signals, core.Signal{
				SymbolID:   symbolID,
				Signal:     core.SignalShort,
				Confidence: 0.7,
				Timestamp:  timestamp,
				StrategyID: 1,
				TargetPrice: currentPrice,
				TargetQuantity: core.QuantityFromFloat(100),
			})
		}
	}

	// RSI signals
	if indicators.RSI > 70 {
		// Overbought
		signals = append(signals, core.Signal{
			SymbolID:   symbolID,
			Signal:     core.SignalShort,
			Confidence: 0.6,
			Timestamp:  timestamp,
			StrategyID: 2,
			TargetPrice: currentPrice,
			TargetQuantity: core.QuantityFromFloat(50),
		})
	} else if indicators.RSI < 30 {
		// Oversold
		signals = append(signals, core.Signal{
			SymbolID:   symbolID,
			Signal:     core.SignalLong,
			Confidence: 0.6,
			Timestamp:  timestamp,
			StrategyID: 2,
			TargetPrice: currentPrice,
			TargetQuantity: core.QuantityFromFloat(50),
		})
	}

	// MACD signals
	if indicators.MACD > 0 && indicators.MACDSignal > 0 {
		if indicators.MACD > indicators.MACDSignal && indicators.MACDHist > 0 {
			signals = append(signals, core.Signal{
				SymbolID:   symbolID,
				Signal:     core.SignalLong,
				Confidence: 0.65,
				Timestamp:  timestamp,
				StrategyID: 3,
				TargetPrice: currentPrice,
				TargetQuantity: core.QuantityFromFloat(75),
			})
		}
	}

	return signals
}

// generateStatisticalSignals generates statistical arbitrage signals
func (se *SignalEngine) generateStatisticalSignals(symbolID core.SymbolID, prices []core.Price) []core.Signal {
	var signals []core.Signal

	if len(prices) < 50 {
		return signals
	}

	// Calculate returns
	returns := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		prev, _ := prices[i-1].Float64()
		curr, _ := prices[i].Float64()
		if prev != 0 {
			returns[i-1] = (curr - prev) / prev
		}
	}

	// Calculate statistical features
	features := se.calculateStatisticalFeatures(returns)

	currentPrice := prices[len(prices)-1]
	timestamp := core.CurrentTimestamp()

	// Momentum signals
	if features.MeanReturn > 0.001 && features.ReturnVolatility < 0.05 {
		// Strong positive momentum with controlled volatility
		signals = append(signals, core.Signal{
			SymbolID:   symbolID,
			Signal:     core.SignalLong,
			Confidence: 0.75,
			Timestamp:  timestamp,
			StrategyID: 4, // Statistical strategy
			TargetPrice: currentPrice,
			TargetQuantity: core.QuantityFromFloat(120),
		})
	} else if features.MeanReturn < -0.001 && features.ReturnVolatility < 0.05 {
		// Strong negative momentum
		signals = append(signals, core.Signal{
			SymbolID:   symbolID,
			Signal:     core.SignalShort,
			Confidence: 0.75,
			Timestamp:  timestamp,
			StrategyID: 4,
			TargetPrice: currentPrice,
			TargetQuantity: core.QuantityFromFloat(120),
		})
	}

	// Mean reversion signals
	recentReturns := returns[len(returns)-10:]
	recentMean := 0.0
	for _, r := range recentReturns {
		recentMean += r
	}
	recentMean /= float64(len(recentReturns))

	if recentMean < features.MeanReturn-2*features.ReturnVolatility {
		// Price significantly below mean - potential reversion up
		signals = append(signals, core.Signal{
			SymbolID:   symbolID,
			Signal:     core.SignalLong,
			Confidence: 0.55,
			Timestamp:  timestamp,
			StrategyID: 5,
			TargetPrice: currentPrice,
			TargetQuantity: core.QuantityFromFloat(80),
		})
	}

	return signals
}

// generateMLSignals generates ML-based signals (simplified placeholder)
func (se *SignalEngine) generateMLSignals(symbolID core.SymbolID, prices []core.Price, volumes []core.Quantity) []core.Signal {
	if len(prices) < 30 {
		return nil
	}

	// Simple momentum-based signal as placeholder for ML
	recentPrices := prices[len(prices)-5:]
	olderPrices := prices[len(prices)-10 : len(prices)-5]

	if len(recentPrices) != 5 || len(olderPrices) != 5 {
		return nil
	}

	recentSum := 0.0
	for _, price := range recentPrices {
		p, _ := price.Float64()
		recentSum += p
	}
	recentAvg := recentSum / 5.0

	olderSum := 0.0
	for _, price := range olderPrices {
		p, _ := price.Float64()
		olderSum += p
	}
	olderAvg := olderSum / 5.0

	momentum := (recentAvg - olderAvg) / olderAvg

	var signalType core.SignalType
	var confidence float32

	if momentum > 0.02 {
		signalType = core.SignalLong
		confidence = 0.8
	} else if momentum < -0.02 {
		signalType = core.SignalShort
		confidence = 0.8
	} else {
		signalType = core.SignalNeutral
		confidence = 0.5
	}

	if signalType != core.SignalNeutral {
		return []core.Signal{{
			SymbolID:     symbolID,
			Signal:       signalType,
			Confidence:   confidence,
			Timestamp:    core.CurrentTimestamp(),
			StrategyID:   100, // ML strategy
			TargetPrice:  prices[len(prices)-1],
			TargetQuantity: core.QuantityFromFloat(50),
		}}
	}

	return nil
}

// filterAndRankSignals filters and ranks signals based on configuration
func (se *SignalEngine) filterAndRankSignals(signals []core.Signal) []core.Signal {
	// Filter by confidence
	filtered := core.FilterSignals(signals, func(s core.Signal) bool {
		return float64(s.Confidence) >= float64(se.config.MinSignalConfidence)
	})

	// Filter by age
	currentTime := core.CurrentTimestamp()
	filtered = core.FilterSignals(filtered, func(s core.Signal) bool {
		age := currentTime - s.Timestamp
		return age <= se.config.MaxSignalAgeNS
	})

	// Group by symbol and limit signals per symbol
	symbolGroups := core.GroupSignalsBySymbol(filtered)
	var finalSignals []core.Signal

	for _, symbolSignals := range symbolGroups {
		// Sort by confidence (highest first)
		// Simple bubble sort for demonstration
		for i := 0; i < len(symbolSignals); i++ {
			for j := i + 1; j < len(symbolSignals); j++ {
				if symbolSignals[i].Confidence < symbolSignals[j].Confidence {
					symbolSignals[i], symbolSignals[j] = symbolSignals[j], symbolSignals[i]
				}
			}
		}

		// Take top N signals
		maxSignals := se.config.MaxSignalsPerSymbol
		if len(symbolSignals) < maxSignals {
			maxSignals = len(symbolSignals)
		}

		finalSignals = append(finalSignals, symbolSignals[:maxSignals]...)
	}

	return finalSignals
}

// calculateStatisticalFeatures calculates statistical features from returns
func (se *SignalEngine) calculateStatisticalFeatures(returns []float64) StatisticalFeatures {
	if len(returns) == 0 {
		return StatisticalFeatures{}
	}

	// Calculate mean
	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	// Calculate variance
	variance := 0.0
	for _, r := range returns {
		diff := r - mean
		variance += diff * diff
	}
	variance /= float64(len(returns))

	volatility := math.Sqrt(variance)

	return StatisticalFeatures{
		MeanReturn:      mean,
		ReturnVolatility: volatility,
	}
}

// StatisticalFeatures holds statistical features
type StatisticalFeatures struct {
	MeanReturn      float64
	ReturnVolatility float64
}

// GetPerformanceCounters returns performance statistics
func (se *SignalEngine) GetPerformanceCounters() *core.PerformanceCounters {
	return se.performanceCounters
}

// ResetPerformanceCounters resets performance counters
func (se *SignalEngine) ResetPerformanceCounters() {
	se.performanceCounters = &core.PerformanceCounters{}
}

// UpdateConfig updates signal engine configuration
func (se *SignalEngine) UpdateConfig(config *SignalConfig) {
	se.config = config
}
