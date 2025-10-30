#!/bin/bash

# Go Trading Engine Build Script
# Builds the high-performance Go trading engine

set -e

echo "🚀 Building Go Trading Engine..."

# Navigate to go engine directory
cd go_engine

# Download dependencies
echo "📦 Downloading dependencies..."
go mod tidy

# Build with optimizations
echo "🔨 Building optimized binary..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o ../bin/go_trading_engine ./cmd/

# Build for testing
echo "🧪 Building test binary..."
go build -o ../bin/go_trading_engine_test ./cmd/

# Run tests
echo "🧪 Running tests..."
go test ./... -v -race

# Run benchmarks if available
echo "📊 Running benchmarks..."
go test -bench=. -benchmem ./... | tee ../logs/go_benchmarks_$(date +%Y%m%d_%H%M%S).log

echo "✅ Go Trading Engine built successfully!"

# Show binary size
if [ -f "../bin/go_trading_engine" ]; then
    echo "📏 Binary size: $(du -h ../bin/go_trading_engine | cut -f1)"
fi

# Show build info
echo ""
echo "🏗️  Build Information:"
echo "  Go version: $(go version)"
echo "  Build time: $(date)"
echo "  Binary location: bin/go_trading_engine"

echo ""
echo "🎯 To run the engine:"
echo "  ./bin/go_trading_engine"

echo ""
echo "🧪 To run tests:"
echo "  go test ./..."

echo ""
echo "📊 To run benchmarks:"
echo "  go test -bench=. ./..."

echo ""
echo "🎉 Go Trading Engine ready for high-performance trading!"
