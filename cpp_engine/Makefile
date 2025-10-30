# Citadel-Style High-Frequency Trading Engine Makefile

# Compiler settings
CXX = g++-11
CXXFLAGS = -std=c++17 -O3 -march=native -flto -DNDEBUG -pthread
CXXFLAGS_DEBUG = -std=c++17 -O0 -g -Wall -Wextra -fsanitize=address
CXXFLAGS_RELWITHDEBINFO = -std=c++17 -O3 -march=native -g -DNDEBUG

# Include and library paths
INCLUDES = -Iinclude -I/usr/local/include -I/opt/homebrew/include
LIBS = -L/usr/lib -L/usr/local/lib -L/opt/homebrew/lib
LDFLAGS = -pthread -lboost_system -lboost_thread -ltbb -lgtest -lgmock -lbenchmark -lglog

# Optional: DPDK support
ifdef USE_DPDK
    CXXFLAGS += -DUSE_DPDK
    LDFLAGS += -lrte_eal -lrte_mempool -lrte_ring -lrte_mbuf -lrte_ethdev
    INCLUDES += -I/usr/local/include/dpdk
endif

# Source files
CORE_SOURCES = \
    src/core/types.cpp \
    src/core/memory_pool.cpp \
    src/core/lock_free_queue.cpp \
    src/core/cache_aligned_allocator.cpp \
    src/core/quantization.cpp

SIGNAL_SOURCES = \
    src/signal_engine/signal_engine.cpp \
    src/signal_engine/technical_indicators.cpp \
    src/signal_engine/statistical_signals.cpp \
    src/signal_engine/ml_signals.cpp

RISK_SOURCES = \
    src/risk_engine/risk_engine.cpp \
    src/risk_engine/var_calculator.cpp \
    src/risk_engine/position_manager.cpp \
    src/risk_engine/stress_tester.cpp

ORDER_SOURCES = \
    src/order_engine/order_engine.cpp \
    src/order_engine/order_router.cpp \
    src/order_engine/fix_protocol.cpp \
    src/order_engine/order_book.cpp

DATA_SOURCES = \
    src/data_engine/data_engine.cpp \
    src/data_engine/market_data_parser.cpp \
    src/data_engine/tick_processor.cpp \
    src/data_engine/book_builder.cpp

EXECUTION_SOURCES = \
    src/execution_engine/execution_engine.cpp \
    src/execution_engine/smart_router.cpp \
    src/execution_engine/latency_arbitrage.cpp

ENGINE_SOURCES = \
    src/engine.cpp \
    $(CORE_SOURCES) \
    $(SIGNAL_SOURCES) \
    $(RISK_SOURCES) \
    $(ORDER_SOURCES) \
    $(DATA_SOURCES) \
    $(EXECUTION_SOURCES)

# Object files
CORE_OBJS = $(CORE_SOURCES:.cpp=.o)
SIGNAL_OBJS = $(SIGNAL_SOURCES:.cpp=.o)
RISK_OBJS = $(RISK_SOURCES:.cpp=.o)
ORDER_OBJS = $(ORDER_SOURCES:.cpp=.o)
DATA_OBJS = $(DATA_SOURCES:.cpp=.o)
EXECUTION_OBJS = $(EXECUTION_SOURCES:.cpp=.o)
ENGINE_OBJS = $(ENGINE_SOURCES:.cpp=.o)

# Build directories
BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
BIN_DIR = $(BUILD_DIR)/bin
TEST_DIR = $(BUILD_DIR)/test
BENCH_DIR = $(BUILD_DIR)/bench

# Targets
.PHONY: all release debug relwithdebinfo clean test benchmark install

all: release

release: CXXFLAGS += $(CXXFLAGS)
release: dirs trading_engine python_bindings tests benchmarks

debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: dirs trading_engine python_bindings tests benchmarks

relwithdebinfo: CXXFLAGS = $(CXXFLAGS_RELWITHDEBINFO)
relwithdebinfo: dirs trading_engine python_bindings tests benchmarks

dirs:
	mkdir -p $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR) $(TEST_DIR) $(BENCH_DIR)

# Core libraries
libcore.a: $(CORE_OBJS)
	ar rcs $(LIB_DIR)/$@ $^

libsignal.a: $(SIGNAL_OBJS) libcore.a
	ar rcs $(LIB_DIR)/$@ $(SIGNAL_OBJS)

librisk.a: $(RISK_OBJS) libcore.a
	ar rcs $(LIB_DIR)/$@ $(RISK_OBJS)

liborder.a: $(ORDER_OBJS) libcore.a
	ar rcs $(LIB_DIR)/$@ $(ORDER_OBJS)

libdata.a: $(DATA_OBJS) libcore.a
	ar rcs $(LIB_DIR)/$@ $(DATA_OBJS)

libexecution.a: $(EXECUTION_OBJS) libcore.a
	ar rcs $(LIB_DIR)/$@ $(EXECUTION_OBJS)

# Main trading engine
trading_engine: libcore.a libsignal.a librisk.a liborder.a libdata.a libexecution.a
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/trading_engine src/engine.cpp \
	    -L$(LIB_DIR) -lsignal -lrisk -lorder -ldata -lexecution -lcore $(LDFLAGS)

# Python bindings
python_bindings: libcore.a libsignal.a librisk.a liborder.a libdata.a libexecution.a
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $(LIB_DIR)/python_bindings.so bindings/python_bindings.cpp \
	    -L$(LIB_DIR) -lsignal -lrisk -lorder -ldata -lexecution -lcore $(LDFLAGS) \
	    `python3-config --includes --ldflags` -lpybind11

# Tests
tests: libcore.a libsignal.a librisk.a liborder.a libdata.a libexecution.a
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/signal_engine_test tests/signal_engine_test.cpp \
	    -L$(LIB_DIR) -lsignal -lcore $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/risk_engine_test tests/risk_engine_test.cpp \
	    -L$(LIB_DIR) -lrisk -lcore $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/order_engine_test tests/order_engine_test.cpp \
	    -L$(LIB_DIR) -lorder -lcore $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/data_engine_test tests/data_engine_test.cpp \
	    -L$(LIB_DIR) -ldata -lcore $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/integration_test tests/integration_test.cpp \
	    -L$(LIB_DIR) -lsignal -lrisk -lorder -ldata -lexecution -lcore $(LDFLAGS)

# Benchmarks
benchmarks: libcore.a libsignal.a librisk.a liborder.a libdata.a libexecution.a
	$(CXX) $(CXXFLAGS) -o $(BENCH_DIR)/signal_benchmark benchmarks/signal_benchmark.cpp \
	    -L$(LIB_DIR) -lsignal -lcore $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o $(BENCH_DIR)/risk_benchmark benchmarks/risk_benchmark.cpp \
	    -L$(LIB_DIR) -lrisk -lcore $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o $(BENCH_DIR)/order_benchmark benchmarks/order_benchmark.cpp \
	    -L$(LIB_DIR) -lorder -lcore $(LDFLAGS)

# Run tests
test: tests
	@echo "Running tests..."
	@./$(TEST_DIR)/signal_engine_test || echo "Signal engine test failed"
	@./$(TEST_DIR)/risk_engine_test || echo "Risk engine test failed"
	@./$(TEST_DIR)/order_engine_test || echo "Order engine test failed"
	@./$(TEST_DIR)/data_engine_test || echo "Data engine test failed"
	@./$(TEST_DIR)/integration_test || echo "Integration test failed"

# Run benchmarks
benchmark: benchmarks
	@echo "Running benchmarks..."
	@./$(BENCH_DIR)/signal_benchmark || echo "Signal benchmark failed"
	@./$(BENCH_DIR)/risk_benchmark || echo "Risk benchmark failed"
	@./$(BENCH_DIR)/order_benchmark || echo "Order benchmark failed"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	find . -name "*.o" -delete
	find . -name "*.a" -delete
	find . -name "*.so" -delete

# Deep clean (including generated files)
distclean: clean
	rm -rf $(BUILD_DIR)

# Installation
install: release
	install -d $(DESTDIR)/usr/local/lib
	install -d $(DESTDIR)/usr/local/include/trading
	install -d $(DESTDIR)/usr/local/bin
	install $(LIB_DIR)/*.a $(DESTDIR)/usr/local/lib/
	install $(LIB_DIR)/*.so $(DESTDIR)/usr/local/lib/
	install $(BIN_DIR)/trading_engine $(DESTDIR)/usr/local/bin/
	cp -r include/* $(DESTDIR)/usr/local/include/trading/

# Development targets
format:
	find include src tests benchmarks -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

lint:
	find include src tests benchmarks -name "*.cpp" -o -name "*.hpp" | xargs clang-tidy -- -Iinclude

# Performance profiling
profile: release
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./$(BIN_DIR)/trading_engine
	kcachegrind callgrind.out

# Code coverage
coverage: debug
	$(CXX) $(CXXFLAGS_DEBUG) --coverage -o $(BIN_DIR)/trading_engine_coverage src/engine.cpp \
	    $(ENGINE_SOURCES) $(LDFLAGS)
	./$(BIN_DIR)/trading_engine_coverage
	lcov --capture --directory . --output-file coverage.info
	lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage.info
	genhtml coverage.info --output-directory coverage_report

# Docker build
docker-build:
	docker build -t citadel-trading-engine .

docker-run:
	docker run -it --rm citadel-trading-engine

# Help target
help:
	@echo "Citadel Trading Engine Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all            - Build everything (release mode)"
	@echo "  release        - Build optimized release version"
	@echo "  debug          - Build debug version with sanitizers"
	@echo "  relwithdebinfo - Build release with debug info"
	@echo "  test           - Run all tests"
	@echo "  benchmark      - Run performance benchmarks"
	@echo "  clean          - Clean build artifacts"
	@echo "  install        - Install to system directories"
	@echo "  format         - Format code with clang-format"
	@echo "  lint           - Run clang-tidy linting"
	@echo "  profile        - Generate performance profile"
	@echo "  coverage       - Generate code coverage report"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run in Docker container"
	@echo ""
	@echo "Configuration variables:"
	@echo "  USE_DPDK=1     - Enable DPDK kernel bypass networking"
	@echo "  CXX=g++-11     - Specify compiler (default: g++-11)"

# Default target
.DEFAULT_GOAL := all
