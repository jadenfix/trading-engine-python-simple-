cmake_minimum_required(VERSION 3.16)
project(TradingEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -march=native -g -DNDEBUG")

# Enable IPO/LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Find dependencies
find_package(Boost 1.70 REQUIRED COMPONENTS system thread)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Optional: DPDK for kernel bypass networking
find_package(PkgConfig)
pkg_check_modules(DPDK libdpdk)

# Include directories
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

# Source files
set(SIGNAL_ENGINE_SOURCES
    src/signal_engine/signal_engine.cpp
    src/signal_engine/technical_indicators.cpp
    src/signal_engine/statistical_signals.cpp
    src/signal_engine/ml_signals.cpp
)

set(RISK_ENGINE_SOURCES
    src/risk_engine/risk_engine.cpp
    src/risk_engine/var_calculator.cpp
    src/risk_engine/position_manager.cpp
    src/risk_engine/stress_tester.cpp
)

set(ORDER_ENGINE_SOURCES
    src/order_engine/order_engine.cpp
    src/order_engine/order_router.cpp
    src/order_engine/fix_protocol.cpp
    src/order_engine/order_book.cpp
)

set(DATA_ENGINE_SOURCES
    src/data_engine/data_engine.cpp
    src/data_engine/market_data_parser.cpp
    src/data_engine/tick_processor.cpp
    src/data_engine/book_builder.cpp
)

set(EXECUTION_ENGINE_SOURCES
    src/execution_engine/execution_engine.cpp
    src/execution_engine/smart_router.cpp
    src/execution_engine/latency_arbitrage.cpp
)

set(CORE_SOURCES
    src/core/types.cpp
    src/core/memory_pool.cpp
    src/core/lock_free_queue.cpp
    src/core/cache_aligned_allocator.cpp
    src/core/quantization.cpp
)

# Create libraries
add_library(signal_engine STATIC ${SIGNAL_ENGINE_SOURCES})
add_library(risk_engine STATIC ${RISK_ENGINE_SOURCES})
add_library(order_engine STATIC ${ORDER_ENGINE_SOURCES})
add_library(data_engine STATIC ${DATA_ENGINE_SOURCES})
add_library(execution_engine STATIC ${EXECUTION_ENGINE_SOURCES})
add_library(core STATIC ${CORE_SOURCES})

# Main engine library
add_library(trading_engine STATIC
    src/engine.cpp
    ${SIGNAL_ENGINE_SOURCES}
    ${RISK_ENGINE_SOURCES}
    ${ORDER_ENGINE_SOURCES}
    ${DATA_ENGINE_SOURCES}
    ${EXECUTION_ENGINE_SOURCES}
    ${CORE_SOURCES}
)

# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(python_bindings
    bindings/python_bindings.cpp
)

# Link libraries
target_link_libraries(signal_engine PRIVATE core)
target_link_libraries(risk_engine PRIVATE core)
target_link_libraries(order_engine PRIVATE core)
target_link_libraries(data_engine PRIVATE core)
target_link_libraries(execution_engine PRIVATE core signal_engine risk_engine order_engine)

target_link_libraries(trading_engine
    PUBLIC
    ${Boost_LIBRARIES}
    tbb
    pthread
)

if(DPDK_FOUND)
    target_link_libraries(data_engine PRIVATE ${DPDK_LIBRARIES})
    target_compile_definitions(data_engine PRIVATE USE_DPDK)
endif()

target_link_libraries(python_bindings PRIVATE trading_engine)

# Tests
enable_testing()

add_executable(signal_engine_test tests/signal_engine_test.cpp)
target_link_libraries(signal_engine_test
    PRIVATE
    signal_engine
    GTest::gtest_main
    benchmark::benchmark
)
add_test(NAME signal_engine_test COMMAND signal_engine_test)

add_executable(risk_engine_test tests/risk_engine_test.cpp)
target_link_libraries(risk_engine_test
    PRIVATE
    risk_engine
    GTest::gtest_main
    benchmark::benchmark
)
add_test(NAME risk_engine_test COMMAND risk_engine_test)

add_executable(order_engine_test tests/order_engine_test.cpp)
target_link_libraries(order_engine_test
    PRIVATE
    order_engine
    GTest::gtest_main
    benchmark::benchmark
)
add_test(NAME order_engine_test COMMAND order_engine_test)

add_executable(data_engine_test tests/data_engine_test.cpp)
target_link_libraries(data_engine_test
    PRIVATE
    data_engine
    GTest::gtest_main
    benchmark::benchmark
)
add_test(NAME data_engine_test COMMAND data_engine_test)

add_executable(integration_test tests/integration_test.cpp)
target_link_libraries(integration_test
    PRIVATE
    trading_engine
    GTest::gtest_main
    benchmark::benchmark
)
add_test(NAME integration_test COMMAND integration_test)

# Benchmarks
add_executable(signal_benchmark benchmarks/signal_benchmark.cpp)
target_link_libraries(signal_benchmark
    PRIVATE
    signal_engine
    benchmark::benchmark
)

add_executable(risk_benchmark benchmarks/risk_benchmark.cpp)
target_link_libraries(risk_benchmark
    PRIVATE
    risk_engine
    benchmark::benchmark
)

add_executable(order_benchmark benchmarks/order_benchmark.cpp)
target_link_libraries(order_benchmark
    PRIVATE
    order_engine
    benchmark::benchmark
)

# Installation
install(TARGETS trading_engine python_bindings
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    TradingEngineConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    cmake/TradingEngineConfig.cmake.in
    TradingEngineConfig.cmake
    INSTALL_DESTINATION lib/cmake/TradingEngine
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TradingEngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TradingEngineConfigVersion.cmake
    DESTINATION lib/cmake/TradingEngine
)
