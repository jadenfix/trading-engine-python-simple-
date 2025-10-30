package risk_engine

import (
	"go-trading-engine/internal/core"
	"go-trading-engine/pkg/types"
	"math"
	"math/rand"
	"time"
)

// RiskEngine provides comprehensive risk management
type RiskEngine struct {
	config              *RiskConfig
	cache               map[core.SymbolID]core.RiskMetrics
	performanceCounters *core.PerformanceCounters
}

// RiskConfig holds risk management configuration
type RiskConfig struct {
	MaxPortfolioVaR   float64 `json:"max_portfolio_var"`
	MaxPositionVaR    float64 `json:"max_position_var"`
	MaxDrawdown       float64 `json:"max_drawdown"`
	MaxLeverage       float64 `json:"max_leverage"`
	MaxConcentration  float64 `json:"max_concentration"`
	MaxPositions      int     `json:"max_positions"`
	KellyFraction     float64 `json:"kelly_fraction"`
}

// NewRiskEngine creates a new risk engine
func NewRiskEngine() *RiskEngine {
	return &RiskEngine{
		config: &RiskConfig{
			MaxPortfolioVaR:  0.05, // 5% VaR limit
			MaxPositionVaR:   0.02, // 2% per position VaR
			MaxDrawdown:      0.10, // 10% max drawdown
			MaxLeverage:      5.0,  // 5x max leverage
			MaxConcentration: 0.25, // 25% max position size
			MaxPositions:     100,  // Max number of positions
			KellyFraction:    0.5,  // Conservative Kelly fraction
		},
		cache:               make(map[core.SymbolID]core.RiskMetrics),
		performanceCounters: &core.PerformanceCounters{},
	}
}

// CheckPortfolioRisk checks portfolio-level risk limits
func (re *RiskEngine) CheckPortfolioRisk(positions []core.Order, marketData []core.MarketData) core.RiskCheckResult {
	start := time.Now()

	// Group market data by symbol
	marketDataMap := make(map[core.SymbolID]core.MarketData)
	for _, data := range marketData {
		marketDataMap[data.SymbolID] = data
	}

	var violations []string
	totalExposure := 0.0

	// Calculate risk for each position
	for _, position := range positions {
		if marketData, exists := marketDataMap[position.SymbolID]; exists {
			riskMetrics, err := re.calculatePositionRisk(&position, &marketData)
			if err != nil {
				violations = append(violations, err.Error())
				continue
			}

			// Check position VaR limit
			if riskMetrics.PositionVaR > re.config.MaxPositionVaR {
				violations = append(violations, "Position VaR exceeds limit")
			}

			// Calculate exposure
			exposure, _ := position.Quantity.Mul(marketData.AskPrice).Float64()
			totalExposure += math.Abs(exposure)

			// Check concentration limit
			concentration := math.Abs(exposure) / totalExposure
			if concentration > re.config.MaxConcentration {
				violations = append(violations, "Position concentration exceeds limit")
			}
		}
	}

	// Check position count limit
	if len(positions) > re.config.MaxPositions {
		violations = append(violations, "Too many positions")
	}

	// Check portfolio VaR (simplified - sum of individual VaRs)
	totalVaR := 0.0
	for _, position := range positions {
		if marketData, exists := marketDataMap[position.SymbolID]; exists {
			riskMetrics, err := re.calculatePositionRisk(&position, &marketData)
			if err == nil {
				totalVaR += riskMetrics.PositionVaR
			}
		}
	}

	if totalVaR > re.config.MaxPortfolioVaR {
		violations = append(violations, "Portfolio VaR exceeds limit")
	}

	processingTime := time.Since(start)
	re.performanceCounters.UpdateRiskLatency(processingTime)
	re.performanceCounters.RiskChecksPerformed++

	return core.RiskCheckResult{
		IsValid:    len(violations) == 0,
		Violations: violations,
	}
}

// calculatePositionRisk calculates risk metrics for a single position
func (re *RiskEngine) calculatePositionRisk(position *core.Order, marketData *core.MarketData) (core.RiskMetrics, error) {
	// Check cache first
	if cached, exists := re.cache[position.SymbolID]; exists {
		// Update with current price
		cached.CurrentPrice = marketData.AskPrice
		unrealizedPnL := core.CalculateUnrealizedPnL(position.Quantity, position.Price, marketData.AskPrice)
		cached.UnrealizedPnL = core.PriceFromFloat(unrealizedPnL)
		cached.Timestamp = core.CurrentTimestamp()
		return cached, nil
	}

	// Generate synthetic returns for VaR calculation (252 trading days)
	returns := re.generateSyntheticReturns(252)

	// Calculate VaR using historical simulation
	var95 := re.calculateHistoricalVaR(returns, 0.95)
	expectedShortfall := re.calculateCVaR(returns, 0.95)

	// Calculate Sharpe ratio (simplified)
	meanReturn := 0.0
	for _, r := range returns {
		meanReturn += r
	}
	meanReturn /= float64(len(returns))

	sharpeRatio := 0.0
	if var95 > 0 {
		sharpeRatio = meanReturn / var95
	}

	riskMetrics := core.RiskMetrics{
		SymbolID:           position.SymbolID,
		CurrentPrice:       marketData.AskPrice,
		PositionSize:       position.Quantity,
		EntryPrice:         position.Price,
		StopLossPrice:      position.Price.Mul(core.DecimalFromFloat(0.95)), // 5% stop loss
		TakeProfitPrice:    position.Price.Mul(core.DecimalFromFloat(1.10)), // 10% take profit
		VaR95:              math.Abs(var95),
		ExpectedShortfall:  math.Abs(expectedShortfall),
		SharpeRatio:        sharpeRatio,
		MaxDrawdown:        0.05, // Placeholder
		Timestamp:          core.CurrentTimestamp(),
		UnrealizedPnL:      core.CalculateUnrealizedPnL(position.Quantity, position.Price, marketData.AskPrice),
		PositionVaR:        core.CalculatePositionVaR(position.Quantity, var95),
		BreachWarning:      false,
	}

	// Cache the result
	re.cache[position.SymbolID] = riskMetrics

	return riskMetrics, nil
}

// CalculateKellyCriterion calculates optimal position sizing using Kelly criterion
func (re *RiskEngine) CalculateKellyCriterion(returns []float64, winRate, winLossRatio *float64) float64 {
	if len(returns) == 0 {
		return 0.0
	}

	// Method 1: Using win rate and win/loss ratio (if provided)
	if winRate != nil && winLossRatio != nil {
		lossRate := 1.0 - *winRate
		return *winRate / *winLossRatio - lossRate
	}

	// Method 2: Using historical returns
	meanReturn := 0.0
	for _, r := range returns {
		meanReturn += r
	}
	meanReturn /= float64(len(returns))

	variance := 0.0
	for _, r := range returns {
		diff := r - meanReturn
		variance += diff * diff
	}
	variance /= float64(len(returns))

	if variance == 0.0 {
		return 0.0
	}

	// Kelly fraction = (expected return) / variance
	kellyFraction := meanReturn / variance

	// Apply conservative fraction
	return kellyFraction * re.config.KellyFraction
}

// CalculateMaxDrawdown calculates maximum drawdown from portfolio values
func (re *RiskEngine) CalculateMaxDrawdown(portfolioValues []float64) float64 {
	if len(portfolioValues) == 0 {
		return 0.0
	}

	maxDrawdown := 0.0
	peak := portfolioValues[0]

	for _, value := range portfolioValues {
		if value > peak {
			peak = value
		}

		drawdown := (peak - value) / peak
		if drawdown > maxDrawdown {
			maxDrawdown = drawdown
		}
	}

	return maxDrawdown
}

// generateSyntheticReturns generates synthetic returns for VaR calculation
func (re *RiskEngine) generateSyntheticReturns(nReturns int) []float64 {
	returns := make([]float64, nReturns)

	// Use fixed seed for reproducible results
	r := rand.New(rand.NewSource(42))

	for i := 0; i < nReturns; i++ {
		// Generate returns with mean 0.01% and std dev 2%
		returns[i] = 0.0001 + r.NormFloat64()*0.02
	}

	return returns
}

// calculateHistoricalVaR calculates Value at Risk using historical simulation
func (re *RiskEngine) calculateHistoricalVaR(returns []float64, confidenceLevel float64) float64 {
	if len(returns) == 0 {
		return 0.0
	}

	// Sort returns in ascending order
	sortedReturns := make([]float64, len(returns))
	copy(sortedReturns, returns)

	for i := 0; i < len(sortedReturns); i++ {
		for j := i + 1; j < len(sortedReturns); j++ {
			if sortedReturns[i] > sortedReturns[j] {
				sortedReturns[i], sortedReturns[j] = sortedReturns[j], sortedReturns[i]
			}
		}
	}

	// Find the return at the confidence level
	index := int((1.0 - confidenceLevel) * float64(len(sortedReturns)))
	if index >= len(sortedReturns) {
		index = len(sortedReturns) - 1
	}

	return sortedReturns[index]
}

// calculateCVaR calculates Conditional VaR (Expected Shortfall)
func (re *RiskEngine) calculateCVaR(returns []float64, confidenceLevel float64) float64 {
	if len(returns) == 0 {
		return 0.0
	}

	// Sort returns in ascending order
	sortedReturns := make([]float64, len(returns))
	copy(sortedReturns, returns)

	for i := 0; i < len(sortedReturns); i++ {
		for j := i + 1; j < len(sortedReturns); j++ {
			if sortedReturns[i] > sortedReturns[j] {
				sortedReturns[i], sortedReturns[j] = sortedReturns[j], sortedReturns[i]
			}
		}
	}

	// Calculate CVaR as average of returns beyond VaR
	varIndex := int((1.0 - confidenceLevel) * float64(len(sortedReturns)))
	if varIndex >= len(sortedReturns) {
		return sortedReturns[len(sortedReturns)-1]
	}

	tailReturns := sortedReturns[:varIndex+1]
	sum := 0.0
	for _, r := range tailReturns {
		sum += r
	}

	return sum / float64(len(tailReturns))
}

// GetPerformanceCounters returns performance statistics
func (re *RiskEngine) GetPerformanceCounters() *core.PerformanceCounters {
	return re.performanceCounters
}

// ResetPerformanceCounters resets performance counters
func (re *RiskEngine) ResetPerformanceCounters() {
	re.performanceCounters = &core.PerformanceCounters{}
}

// ClearCache clears the risk metrics cache
func (re *RiskEngine) ClearCache() {
	re.cache = make(map[core.SymbolID]core.RiskMetrics)
}

// UpdateConfig updates risk engine configuration
func (re *RiskEngine) UpdateConfig(config *RiskConfig) {
	re.config = config
}
