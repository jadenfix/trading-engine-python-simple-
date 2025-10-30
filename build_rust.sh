#!/bin/bash

# Rust Trading Engine Build Script
# This script builds the Rust trading engine with optimal performance settings

set -e

echo "🚀 Building Rust Trading Engine..."

# Navigate to rust engine directory
cd rust_engine

# Clean previous build
echo "🧹 Cleaning previous build..."
cargo clean

# Build with release optimizations
echo "🔨 Building release version..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort -C overflow-checks=false" \
cargo build --release --features python

# Build Python wheel if maturin is available
if command -v maturin &> /dev/null; then
    echo "📦 Building Python wheel..."
    maturin build --release --strip
fi

# Run tests
echo "🧪 Running tests..."
cargo test --release

# Run benchmarks if criterion is available
if cargo bench --help &> /dev/null; then
    echo "📊 Running benchmarks..."
    cargo bench
fi

echo "✅ Build complete!"

# Show binary size
if [ -f "target/release/librust_trading_engine.dylib" ]; then
    echo "📏 Library size: $(du -h target/release/librust_trading_engine.dylib | cut -f1)"
elif [ -f "target/release/librust_trading_engine.so" ]; then
    echo "📏 Library size: $(du -h target/release/librust_trading_engine.so | cut -f1)"
fi

echo "🎉 Rust Trading Engine built successfully!"
echo ""
echo "To use from Python:"
echo "  export PYTHONPATH=\$PYTHONPATH:$(pwd)/target/release"
echo "  python -c \"import rust_trading_engine; print('Ready!')\""
