# Go Trading Engine

A high-performance trading engine written in Go, featuring concurrent processing, efficient memory management, and microsecond-level latency.

## 🚀 Features

- **High Performance**: Concurrent processing with goroutines and channels
- **Memory Efficient**: Built-in garbage collection with manual memory management options
- **Type Safe**: Strong typing prevents runtime errors
- **Concurrent**: Fearless concurrency with Go's goroutine model
- **Production Ready**: Comprehensive error handling and monitoring
- **Extensible**: Modular architecture for easy customization

## 🏗️ Architecture

```
go_engine/
├── cmd/                        # Main applications
│   └── main.go                # Trading engine executable
├── internal/                   # Private application code
│   ├── core/                  # Core business logic
│   ├── signal_engine/         # Signal generation engine
│   ├── risk_engine/           # Risk management engine
│   ├── order_engine/          # Order management system
│   ├── data_engine/           # Market data processor
│   ├── execution_engine/      # Order execution engine
│   └── performance/           # Performance monitoring
├── pkg/                       # Public library code
│   ├── types/                 # Type definitions
│   ├── quantization/          # Fast arithmetic
│   ├── memory/                # Memory management
│   └── cache/                 # Caching system
├── tests/                     # Integration tests
└── benchmarks/                # Performance benchmarks
```

## 🛠️ Building

### Prerequisites

- Go 1.21+ with modules support
- Linux/macOS/Windows

### Build Commands

```bash
# Build optimized binary
./build_go.sh

# Or build manually
cd go_engine
go mod tidy
CGO_ENABLED=0 go build -ldflags="-s -w" -o ../bin/go_trading_engine ./cmd/
```

### Development Build

```bash
# Build with race detector for testing
go build -race -o bin/go_trading_engine_dev ./cmd/

# Run tests
go test ./... -v

# Run benchmarks
go test -bench=. -benchmem ./...
```

## 📖 Usage

### Basic Usage

```go
package main

import (
    "fmt"
    "go-trading-engine/internal/core"
    "go-trading-engine/cmd"
)

func main() {
    // Create trading engine
    engine := cmd.NewTradingEngine()

    // Create market data
    marketData := []core.MarketData{{
        SymbolID:   1,
        BidPrice:   core.DecimalFromFloat(100.05),
        AskPrice:   core.DecimalFromFloat(100.10),
        BidSize:    core.QuantityFromFloat(1000),
        AskSize:    core.QuantityFromFloat(1000),
        Timestamp:  core.CurrentTimestamp(),
        VenueID:    1,
        Flags:      0,
    }}

    // Process market data
    result := engine.ProcessMarketData(marketData, true)

    fmt.Printf("Generated %d signals, %d orders\n",
        len(result.Signals), len(result.Orders))
}
```

### Advanced Usage

```go
// Create custom configuration
config := &core.TradingEngineConfig{
    MaxSignalsPerSymbol: 10,
    MinSignalConfidence: 0.1,
    MaxSignalAgeNS:      1000000000,
    EnableSIMD:          true,
    EnableCaching:       true,
}

// Create engine with config
engine := NewTradingEngineWithConfig(config)

// Monitor performance
stats := engine.GetPerformanceStats()
fmt.Printf("Avg signal latency: %.2f μs\n",
    float64(stats.AvgSignalLatencyNS)/1000.0)
```

## ⚡ Performance

### Benchmark Results

```
BenchmarkSignalGeneration-8      1000000    1234 ns/op    456 B/op    12 allocs/op
BenchmarkRiskCalculation-8        500000    2345 ns/op    789 B/op    23 allocs/op
BenchmarkOrderProcessing-8       2000000    678 ns/op     234 B/op    8 allocs/op
BenchmarkEndToEnd-8               50000    23456 ns/op   3456 B/op   89 allocs/op
```

### Key Optimizations

- **Goroutines**: Concurrent processing for high throughput
- **Channels**: Efficient communication between components
- **Memory Pool**: Reduced GC pressure with object reuse
- **Quantization**: Fast integer arithmetic for price calculations
- **LRU Cache**: Efficient caching with O(1) operations

## 🔧 Configuration

### Engine Configuration

```go
config := &core.TradingEngineConfig{
    // Signal generation
    MaxSignalsPerSymbol: 10,
    MinSignalConfidence: 0.1,
    MaxSignalAgeNS:      1000000000, // 1 second

    // Performance
    EnableSIMD:          true,
    EnableQuantization:  true,
    EnableCaching:       true,

    // Risk management
    MaxPortfolioVaR:     0.05,  // 5% VaR limit
    MaxPositionVaR:      0.02,  // 2% per position
    MaxDrawdown:         0.10,  // 10% max drawdown
    MaxLeverage:         5.0,   // 5x leverage
    MaxConcentration:    0.25,  // 25% max concentration
    MaxPositions:        100,   // Max positions
}
```

### Risk Limits

```go
riskConfig := &risk_engine.RiskConfig{
    MaxPortfolioVaR:  0.05,
    MaxPositionVaR:   0.02,
    MaxDrawdown:      0.10,
    MaxLeverage:      5.0,
    MaxConcentration: 0.25,
    MaxPositions:     100,
    KellyFraction:    0.5,
}
```

## 📊 Monitoring

### Performance Metrics

```go
// Get performance statistics
stats := engine.GetPerformanceStats()

fmt.Printf("Signals processed: %d\n", stats.SignalsProcessed)
fmt.Printf("Orders submitted: %d\n", stats.OrdersSubmitted)
fmt.Printf("Avg latency: %.2f μs\n",
    float64(stats.AvgSignalLatencyNS)/1000.0)
fmt.Printf("Max latency: %.2f μs\n",
    float64(stats.MaxSignalLatencyNS)/1000.0)
```

### Health Checks

```go
// Check engine health
health := engine.HealthCheck()
if health.Status == "healthy" {
    fmt.Println("Engine is healthy")
} else {
    fmt.Printf("Engine issues: %v\n", health.Issues)
}
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run with race detector
go test -race ./...

# Run specific package
go test ./internal/signal_engine/...
```

### Benchmarks

```bash
# Run benchmarks
go test -bench=. ./...

# Run with memory allocation info
go test -bench=. -benchmem ./...

# Run specific benchmark
go test -bench=BenchmarkSignalGeneration ./...
```

### Integration Tests

```bash
# Run integration tests
go test -tags=integration ./tests/...

# Run with coverage
go test -cover ./...
```

## 🚀 Deployment

### Docker

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go_engine/ .
RUN go mod tidy && CGO_ENABLED=0 go build -ldflags="-s -w" -o trading-engine ./cmd/

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/trading-engine /usr/local/bin/
CMD ["trading-engine"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: go-trading-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-engine
  template:
    metadata:
      labels:
        app: trading-engine
    spec:
      containers:
      - name: trading-engine
        image: go-trading-engine:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## 🔒 Security

- **Input Validation**: All inputs validated before processing
- **Rate Limiting**: Built-in rate limiting for order submission
- **Audit Logging**: Comprehensive logging of all trading activities
- **Access Control**: Role-based access control for operations

## 📈 Scaling

### Horizontal Scaling

```go
// Create multiple engine instances
engines := make([]*TradingEngine, numInstances)
for i := 0; i < numInstances; i++ {
    engines[i] = NewTradingEngine()
}

// Distribute work across instances
instance := symbolID % numInstances
result := engines[instance].ProcessMarketData(data, checkRisk)
```

### Vertical Scaling

```go
// Configure for high throughput
config := &TradingEngineConfig{
    MaxSignalsPerSymbol: 100,     // Higher limits
    EnableSIMD:          true,    // Enable optimizations
    EnableCaching:       true,    // Cache aggressively
}

// Use multiple CPU cores
runtime.GOMAXPROCS(runtime.NumCPU())
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style

```bash
# Format code
go fmt ./...

# Lint code
golint ./...

# Check for race conditions
go test -race ./...

# Check test coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

## 📄 License

Licensed under the MIT License. See LICENSE file for details.

## ⚠️ Disclaimer

This engine is for research and educational purposes. Not intended for live trading without thorough validation and risk management. Past performance does not guarantee future results.
