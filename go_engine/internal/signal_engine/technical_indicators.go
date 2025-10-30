package signal_engine

import (
	"go-trading-engine/pkg/types"
	"math"
)

// IndicatorValues holds calculated technical indicators
type IndicatorValues struct {
	SMA20     float64 `json:"sma_20"`
	SMA50     float64 `json:"sma_50"`
	EMA12     float64 `json:"ema_12"`
	EMA26     float64 `json:"ema_26"`
	RSI       float64 `json:"rsi"`
	MACD      float64 `json:"macd"`
	MACDHist  float64 `json:"macd_hist"`
	MACDSignal float64 `json:"macd_signal"`
	BBUpper   float64 `json:"bb_upper"`
	BBLower   float64 `json:"bb_lower"`
	BBMiddle  float64 `json:"bb_middle"`
	ATR       float64 `json:"atr"`
	StochK    float64 `json:"stoch_k"`
	StochD    float64 `json:"stoch_d"`
}

// TechnicalIndicators provides technical indicator calculations
type TechnicalIndicators struct {
	quantizer *types.PriceQuantizer
}

// NewTechnicalIndicators creates a new technical indicators calculator
func NewTechnicalIndicators() *TechnicalIndicators {
	return &TechnicalIndicators{
		quantizer: &types.PriceQuantizer{ScaleFactor: 100},
	}
}

// CalculateIndicators calculates all technical indicators
func (ti *TechnicalIndicators) CalculateIndicators(prices []types.Price, volumes []types.Quantity) IndicatorValues {
	if len(prices) == 0 {
		return IndicatorValues{}
	}

	var indicators IndicatorValues

	// Convert prices to floats for calculation
	priceFloats := make([]float64, len(prices))
	for i, price := range prices {
		priceFloats[i], _ = price.Float64()
	}

	// Simple Moving Averages
	indicators.SMA20 = ti.calculateSMA(priceFloats, 20)
	indicators.SMA50 = ti.calculateSMA(priceFloats, 50)

	// Exponential Moving Averages
	indicators.EMA12 = ti.calculateEMA(priceFloats, 12)
	indicators.EMA26 = ti.calculateEMA(priceFloats, 26)

	// RSI
	indicators.RSI = ti.calculateRSI(priceFloats, 14)

	// MACD
	indicators.MACD, indicators.MACDSignal, indicators.MACDHist = ti.calculateMACD(priceFloats, 12, 26, 9)

	// Bollinger Bands
	indicators.BBUpper, indicators.BBLower, indicators.BBMiddle = ti.calculateBollingerBands(priceFloats, 20, 2.0)

	// ATR
	indicators.ATR = ti.calculateATR(prices, 14)

	// Stochastic Oscillator
	indicators.StochK, indicators.StochD = ti.calculateStochastic(prices, 14, 3)

	return indicators
}

// calculateSMA calculates Simple Moving Average
func (ti *TechnicalIndicators) calculateSMA(prices []float64, period int) float64 {
	if len(prices) < period {
		return 0.0
	}

	sum := 0.0
	for i := len(prices) - period; i < len(prices); i++ {
		sum += prices[i]
	}
	return sum / float64(period)
}

// calculateEMA calculates Exponential Moving Average
func (ti *TechnicalIndicators) calculateEMA(prices []float64, period int) float64 {
	if len(prices) == 0 {
		return 0.0
	}

	multiplier := 2.0 / (float64(period) + 1.0)
	ema := prices[0]

	for i := 1; i < len(prices); i++ {
		ema = (prices[i] * multiplier) + (ema * (1 - multiplier))
	}

	return ema
}

// calculateRSI calculates Relative Strength Index
func (ti *TechnicalIndicators) calculateRSI(prices []float64, period int) float64 {
	if len(prices) < period+1 {
		return 50.0 // Neutral RSI
	}

	gains := make([]float64, 0, len(prices)-1)
	losses := make([]float64, 0, len(prices)-1)

	// Calculate price changes
	for i := 1; i < len(prices); i++ {
		change := prices[i] - prices[i-1]
		if change > 0 {
			gains = append(gains, change)
			losses = append(losses, 0)
		} else {
			gains = append(gains, 0)
			losses = append(losses, -change)
		}
	}

	// Calculate average gains and losses
	avgGain := ti.calculateSMA(gains, period)
	avgLoss := ti.calculateSMA(losses, period)

	if avgLoss == 0.0 {
		return 100.0 // Extremely overbought
	}

	rs := avgGain / avgLoss
	return 100.0 - (100.0 / (1.0 + rs))
}

// calculateMACD calculates MACD (Moving Average Convergence Divergence)
func (ti *TechnicalIndicators) calculateMACD(prices []float64, fastPeriod, slowPeriod, signalPeriod int) (float64, float64, float64) {
	if len(prices) < slowPeriod {
		return 0.0, 0.0, 0.0
	}

	emaFast := ti.calculateEMA(prices, fastPeriod)
	emaSlow := ti.calculateEMA(prices, slowPeriod)

	macd := emaFast - emaSlow
	signal := ti.calculateEMA([]float64{macd}, signalPeriod)
	histogram := macd - signal

	return macd, signal, histogram
}

// calculateBollingerBands calculates Bollinger Bands
func (ti *TechnicalIndicators) calculateBollingerBands(prices []float64, period int, stdDevMultiplier float64) (float64, float64, float64) {
	if len(prices) < period {
		return 0.0, 0.0, 0.0
	}

	sma := ti.calculateSMA(prices, period)

	// Calculate standard deviation
	sumSquares := 0.0
	for i := len(prices) - period; i < len(prices); i++ {
		diff := prices[i] - sma
		sumSquares += diff * diff
	}
	variance := sumSquares / float64(period)
	stdDev := math.Sqrt(variance)

	upper := sma + stdDevMultiplier*stdDev
	lower := sma - stdDevMultiplier*stdDev

	return upper, lower, sma
}

// calculateATR calculates Average True Range
func (ti *TechnicalIndicators) calculateATR(prices []types.Price, period int) float64 {
	if len(prices) < period+1 {
		return 0.0
	}

	trueRanges := make([]float64, 0, len(prices)-1)

	for i := 1; i < len(prices); i++ {
		high, _ := prices[i].Float64()
		low, _ := prices[i].Float64()  // Using price as both high and low approximation
		prevClose, _ := prices[i-1].Float64()

		tr1 := high - low
		tr2 := math.Abs(high - prevClose)
		tr3 := math.Abs(low - prevClose)

		trueRange := math.Max(tr1, math.Max(tr2, tr3))
		trueRanges = append(trueRanges, trueRange)
	}

	return ti.calculateSMA(trueRanges, period)
}

// calculateStochastic calculates Stochastic Oscillator
func (ti *TechnicalIndicators) calculateStochastic(prices []types.Price, kPeriod, dPeriod int) (float64, float64) {
	if len(prices) < kPeriod {
		return 50.0, 50.0
	}

	priceFloats := make([]float64, len(prices))
	for i, price := range prices {
		priceFloats[i], _ = price.Float64()
	}

	// Calculate %K
	kValues := make([]float64, 0, len(prices)-kPeriod+1)

	for i := kPeriod - 1; i < len(priceFloats); i++ {
		window := priceFloats[i-kPeriod+1 : i+1]
		highest := window[0]
		lowest := window[0]

		for _, price := range window {
			if price > highest {
				highest = price
			}
			if price < lowest {
				lowest = price
			}
		}

		current := priceFloats[i]

		var k float64
		if highest != lowest {
			k = 100.0 * (current - lowest) / (highest - lowest)
		} else {
			k = 50.0
		}

		kValues = append(kValues, k)
	}

	// Calculate %D (SMA of %K)
	d := ti.calculateSMA(kValues, dPeriod)

	// Return most recent %K and %D
	if len(kValues) == 0 {
		return 50.0, 50.0
	}

	return kValues[len(kValues)-1], d
}

// CalculateROC calculates Rate of Change
func (ti *TechnicalIndicators) CalculateROC(prices []float64, period int) float64 {
	if len(prices) < period+1 {
		return 0.0
	}

	current := prices[len(prices)-1]
	past := prices[len(prices)-1-period]

	if past == 0.0 {
		return 0.0
	}

	return ((current - past) / past) * 100.0
}

// CalculateWilliamsR calculates Williams %R
func (ti *TechnicalIndicators) CalculateWilliamsR(prices []types.Price, period int) float64 {
	if len(prices) < period {
		return -50.0
	}

	priceFloats := make([]float64, len(prices))
	for i, price := range prices {
		priceFloats[i], _ = price.Float64()
	}

	window := priceFloats[len(priceFloats)-period:]
	highest := window[0]
	lowest := window[0]

	for _, price := range window {
		if price > highest {
			highest = price
		}
		if price < lowest {
			lowest = price
		}
	}

	current := priceFloats[len(priceFloats)-1]

	if highest == lowest {
		return -50.0
	}

	return -100.0 * (highest - current) / (highest - lowest)
}

// CalculateCCI calculates Commodity Channel Index
func (ti *TechnicalIndicators) CalculateCCI(prices []types.Price, period int) float64 {
	if len(prices) < period {
		return 0.0
	}

	priceFloats := make([]float64, len(prices))
	for i, price := range prices {
		priceFloats[i], _ = price.Float64()
	}

	// Calculate Typical Price (SMA of High, Low, Close)
	tpValues := priceFloats[len(priceFloats)-period:] // Approximation using close prices
	smaTP := ti.calculateSMA(tpValues, period)

	// Calculate Mean Deviation
	meanDeviation := 0.0
	for _, tp := range tpValues {
		meanDeviation += math.Abs(tp - smaTP)
	}
	meanDeviation /= float64(period)

	if meanDeviation == 0.0 {
		return 0.0
	}

	currentTP := tpValues[len(tpValues)-1]
	return (currentTP - smaTP) / (0.015 * meanDeviation)
}
