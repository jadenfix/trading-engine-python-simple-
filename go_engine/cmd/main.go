package main

import (
	"fmt"
	"log"
	"time"

	"go-trading-engine/internal/core"
	"go-trading-engine/internal/signal_engine"
	"go-trading-engine/internal/risk_engine"
	"go-trading-engine/internal/order_engine"
	"go-trading-engine/internal/data_engine"
	"go-trading-engine/internal/execution_engine"
	"go-trading-engine/internal/performance"
)

type TradingEngine struct {
	signalEngine   *signal_engine.SignalEngine
	riskEngine     *risk_engine.RiskEngine
	orderEngine    *order_engine.OrderEngine
	dataEngine     *data_engine.DataEngine
	executionEngine *execution_engine.ExecutionEngine
	performance    *performance.PerformanceMonitor
}

func NewTradingEngine() *TradingEngine {
	return &TradingEngine{
		signalEngine:    signal_engine.NewSignalEngine(),
		riskEngine:      risk_engine.NewRiskEngine(),
		orderEngine:     order_engine.NewOrderEngine(),
		dataEngine:      data_engine.NewDataEngine(),
		executionEngine: execution_engine.NewExecutionEngine(),
		performance:     performance.NewPerformanceMonitor(),
	}
}

func (te *TradingEngine) ProcessMarketData(marketData []core.MarketData, checkRisk bool) core.TradingResult {
	start := time.Now()

	// Update performance counter
	te.performance.IncrementMarketDataProcessed(len(marketData))

	// Process market data
	processedData := te.dataEngine.ProcessMarketData(marketData)

	// Generate signals
	signalStart := time.Now()
	signals := te.signalEngine.GenerateSignals(&processedData)
	signalTime := time.Since(signalStart)
	te.performance.UpdateSignalLatency(signalTime)
	te.performance.AddSignalsProcessed(len(signals))

	// Apply risk management if requested
	var orders []core.Order
	if checkRisk {
		riskStart := time.Now()

		// Create positions from signals for risk checking
		positions := make([]core.Order, len(signals))
		for i, signal := range signals {
			positions[i] = core.Order{
				SymbolID: signal.SymbolID,
				Quantity: signal.TargetQuantity,
				Price:    signal.TargetPrice,
			}
		}

		// Check risk limits
		riskResult := te.riskEngine.CheckPortfolioRisk(positions, processedData)
		riskTime := time.Since(riskStart)
		te.performance.UpdateRiskLatency(riskTime)

		if riskResult.IsValid {
			// Generate orders from signals
			orders = make([]core.Order, len(signals))
			for i, signal := range signals {
				var side core.OrderSide
				switch signal.Signal {
				case core.SignalLong, core.SignalExitShort:
					side = core.OrderSideBuy
				case core.SignalShort, core.SignalExitLong:
					side = core.OrderSideSell
				default:
					side = core.OrderSideBuy
				}

				orders[i] = core.Order{
					SymbolID: signal.SymbolID,
					OrderType: core.OrderTypeMarket,
					Side:      side,
					Price:     signal.TargetPrice,
					Quantity:  signal.TargetQuantity,
					CreateTime: core.CurrentTimestamp(),
					UpdateTime: core.CurrentTimestamp(),
				}
			}
		}
		// If risk check fails, orders remains empty
	} else {
		// Generate orders without risk checking
		orders = make([]core.Order, len(signals))
		for i, signal := range signals {
			var side core.OrderSide
			switch signal.Signal {
			case core.SignalLong, core.SignalExitShort:
				side = core.OrderSideBuy
			case core.SignalShort, core.SignalExitLong:
				side = core.OrderSideSell
			default:
				side = core.OrderSideBuy
			}

			orders[i] = core.Order{
				SymbolID: signal.SymbolID,
				OrderType: core.OrderTypeMarket,
				Side:      side,
				Price:     signal.TargetPrice,
				Quantity:  signal.TargetQuantity,
				CreateTime: core.CurrentTimestamp(),
				UpdateTime: core.CurrentTimestamp(),
			}
		}
	}

	te.performance.AddOrdersSubmitted(len(orders))
	totalTime := time.Since(start)

	return core.TradingResult{
		Signals:         signals,
		Orders:          orders,
		ProcessingTimeNS: totalTime.Nanoseconds(),
		MarketDataProcessed: len(processedData),
	}
}

func (te *TradingEngine) GetPerformanceStats() *performance.PerformanceCounters {
	return te.performance.GetCounters()
}

func (te *TradingEngine) ResetPerformanceStats() {
	te.performance.Reset()
}

func main() {
	fmt.Println("🚀 Go Trading Engine - High Performance Trading System")
	fmt.Println("==================================================")

	// Create trading engine
	engine := NewTradingEngine()

	// Create sample market data
	marketData := []core.MarketData{
		{
			SymbolID:   1,
			BidPrice:   core.DecimalFromFloat(100.05),
			AskPrice:   core.DecimalFromFloat(100.10),
			BidSize:    core.DecimalFromFloat(1000),
			AskSize:    core.DecimalFromFloat(1000),
			Timestamp:  core.CurrentTimestamp(),
			VenueID:    1,
			Flags:      0,
		},
		{
			SymbolID:   2,
			BidPrice:   core.DecimalFromFloat(200.15),
			AskPrice:   core.DecimalFromFloat(200.20),
			BidSize:    core.DecimalFromFloat(500),
			AskSize:    core.DecimalFromFloat(500),
			Timestamp:  core.CurrentTimestamp(),
			VenueID:    1,
			Flags:      0,
		},
	}

	fmt.Printf("📊 Processing %d market data points...\n", len(marketData))

	// Process market data
	result := engine.ProcessMarketData(marketData, true)

	// Display results
	fmt.Printf("✅ Processing completed in %.2f μs\n", float64(result.ProcessingTimeNS)/1000.0)
	fmt.Printf("📈 Signals generated: %d\n", len(result.Signals))
	fmt.Printf("📋 Orders created: %d\n", len(result.Orders))

	// Display performance stats
	stats := engine.GetPerformanceStats()
	fmt.Println("\n📊 Performance Statistics:")
	fmt.Printf("   Market data processed: %d\n", stats.MarketDataProcessed)
	fmt.Printf("   Signals processed: %d\n", stats.SignalsProcessed)
	fmt.Printf("   Orders submitted: %d\n", stats.OrdersSubmitted)
	fmt.Printf("   Average signal latency: %.2f μs\n", float64(stats.AvgSignalLatencyNS)/1000.0)
	fmt.Printf("   Average risk latency: %.2f μs\n", float64(stats.AvgRiskLatencyNS)/1000.0)

	fmt.Println("\n🎉 Go Trading Engine ready for high-performance trading!")
}
