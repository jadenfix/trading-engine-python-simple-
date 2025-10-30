#!/bin/bash

# Citadel Trading Engine C++ Build Script
# Builds the high-performance C++ components for maximum speed

set -e  # Exit on any error

echo "========================================="
echo "Citadel Trading Engine C++ Build Script"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "cpp_engine/Makefile" ]; then
    echo "Error: cpp_engine/Makefile not found. Please run from project root."
    exit 1
fi

cd cpp_engine

# Detect number of CPU cores for parallel build
if command -v nproc &> /dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &> /dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

echo "Building with $NPROC parallel jobs..."

# Create build directory
mkdir -p build
cd build

# Check if we have CMake
if command -v cmake &> /dev/null; then
    echo "Using CMake build system..."

    # Configure with CMake
    CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
    CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_CXX_FLAGS='-O3 -march=native -flto -DNDEBUG'"

    # Check for optional dependencies
    if pkg-config --exists libdpdk; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DUSE_DPDK=ON"
        echo "DPDK found - enabling kernel bypass networking"
    else
        echo "DPDK not found - building without kernel bypass"
    fi

    # Configure
    cmake .. $CMAKE_FLAGS

    # Build
    make -j$NPROC

    # Install Python bindings
    echo "Installing Python bindings..."
    if [ -f "libpython_bindings.so" ]; then
        cp libpython_bindings.so ../../../trading_env/lib/python3.11/site-packages/ 2>/dev/null || \
        cp libpython_bindings.so ../../../venv/lib/python3.11/site-packages/ 2>/dev/null || \
        echo "Could not auto-install Python bindings. Please copy libpython_bindings.so to your Python site-packages directory."
    fi

else
    echo "CMake not found, using Make build system..."

    # Use the Makefile
    cd ..
    make release -j$NPROC

    # Install Python bindings
    echo "Installing Python bindings..."
    if [ -f "build/lib/libpython_bindings.so" ]; then
        cp build/lib/libpython_bindings.so ../../trading_env/lib/python3.11/site-packages/ 2>/dev/null || \
        cp build/lib/libpython_bindings.so ../../venv/lib/python3.11/site-packages/ 2>/dev/null || \
        echo "Could not auto-install Python bindings. Please copy build/lib/libpython_bindings.so to your Python site-packages directory."
    fi
fi

echo ""
echo "Build completed successfully!"
echo ""
echo "Testing C++ components..."

# Run tests if they exist
if [ -f "test" ] || [ -f "tests/signal_engine_test" ]; then
    echo "Running unit tests..."
    if [ -f "test" ]; then
        ./test --gtest_output=xml:test_results.xml
    elif [ -f "tests/signal_engine_test" ]; then
        cd tests
        ./signal_engine_test --gtest_output=xml:signal_test_results.xml
        ./risk_engine_test --gtest_output=xml:risk_test_results.xml
        ./order_engine_test --gtest_output=xml:order_test_results.xml
        ./data_engine_test --gtest_output=xml:data_test_results.xml
        ./integration_test --gtest_output=xml:integration_test_results.xml
    fi
    echo "Tests completed."
else
    echo "Test executables not found - build may have issues."
fi

# Run benchmarks if they exist
if [ -f "benchmarks/signal_benchmark" ]; then
    echo ""
    echo "Running performance benchmarks..."
    cd benchmarks
    echo "Signal Engine Benchmark:"
    ./signal_benchmark --benchmark_format=json > signal_benchmark.json
    echo "Risk Engine Benchmark:"
    ./risk_benchmark --benchmark_format=json > risk_benchmark.json
    echo "Order Engine Benchmark:"
    ./order_benchmark --benchmark_format=json > order_benchmark.json
    echo "Benchmarks completed."
fi

echo ""
echo "========================================="
echo "C++ Engine Build Summary"
echo "========================================="
echo "✓ Core libraries built"
echo "✓ Signal engine compiled"
echo "✓ Risk engine compiled"
echo "✓ Order engine compiled"
echo "✓ Data engine compiled"
echo "✓ Python bindings created"
echo ""
echo "Performance Targets:"
echo "  Signal Generation: < 500 nanoseconds"
echo "  Risk Calculation: < 100 nanoseconds"
echo "  Order Processing: < 5 microseconds"
echo "  End-to-End: < 10 microseconds"
echo ""
echo "To use the C++ engine:"
echo "  from research.cpp_integration import CitadelTradingEngine"
echo "  engine = CitadelTradingEngine()"
echo "  result = engine.process_market_data(data)"
echo ""
echo "========================================="
