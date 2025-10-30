package execution_engine

import (
	"go-trading-engine/internal/core"
	"go-trading-engine/pkg/types"
	"sync"
	"time"
)

// ExecutionEngine handles order execution and fills
type ExecutionEngine struct {
	executionQueue      []*core.Order
	executionResults    []*ExecutionResult
	maxQueueSize        int
	performanceCounters *core.PerformanceCounters
	mutex               sync.RWMutex
}

// ExecutionResult represents the result of order execution
type ExecutionResult struct {
	OrderID         core.OrderID    `json:"order_id"`
	ExecutedQuantity core.Quantity  `json:"executed_quantity"`
	ExecutedPrice   core.Price      `json:"executed_price"`
	ExecutionTime   core.Timestamp  `json:"execution_time"`
	VenueID         core.VenueID    `json:"venue_id"`
	Status          ExecutionStatus `json:"status"`
}

// ExecutionStatus represents execution status
type ExecutionStatus int

const (
	ExecutionStatusPartial ExecutionStatus = iota
	ExecutionStatusFilled
	ExecutionStatusRejected
	ExecutionStatusCancelled
)

// NewExecutionEngine creates a new execution engine
func NewExecutionEngine() *ExecutionEngine {
	return &ExecutionEngine{
		executionQueue:      make([]*core.Order, 0, 1000),
		executionResults:    make([]*ExecutionResult, 0, 10000),
		maxQueueSize:        10000,
		performanceCounters: &core.PerformanceCounters{},
	}
}

// QueueOrder adds an order to the execution queue
func (ee *ExecutionEngine) QueueOrder(order *core.Order) bool {
	ee.mutex.Lock()
	defer ee.mutex.Unlock()

	if len(ee.executionQueue) >= ee.maxQueueSize {
		return false // Queue full
	}

	// Create a copy of the order for the queue
	orderCopy := *order
	ee.executionQueue = append(ee.executionQueue, &orderCopy)

	return true
}

// ExecuteOrders processes queued orders (simplified implementation)
func (ee *ExecutionEngine) ExecuteOrders() []*ExecutionResult {
	ee.mutex.Lock()
	defer ee.mutex.Unlock()

	results := make([]*ExecutionResult, 0, len(ee.executionQueue))

	// Process each order in the queue
	for _, order := range ee.executionQueue {
		result := ee.executeOrder(order)
		results = append(results, result)
		ee.executionResults = append(ee.executionResults, result)
	}

	// Clear the queue
	ee.executionQueue = ee.executionQueue[:0]

	ee.performanceCounters.OrdersExecuted += int64(len(results))

	return results
}

// executeOrder executes a single order (simplified)
func (ee *ExecutionEngine) executeOrder(order *core.Order) *ExecutionResult {
	start := time.Now()

	// Simulate execution latency
	time.Sleep(time.Microsecond * 50) // Simulate 50μs execution time

	executionTime := core.CurrentTimestamp()

	var executedQuantity core.Quantity
	var executedPrice core.Price
	var status ExecutionStatus

	switch order.OrderType {
	case core.OrderTypeMarket:
		// Market orders fill immediately at market price
		executedQuantity = order.Quantity
		executedPrice = order.Price // In reality, this would be the market price
		status = ExecutionStatusFilled

	case core.OrderTypeLimit:
		// Limit orders - simplified: assume they get filled
		executedQuantity = order.Quantity
		executedPrice = order.Price
		status = ExecutionStatusFilled

	default:
		// Unknown order type - reject
		executedQuantity = core.QuantityFromFloat(0)
		executedPrice = core.PriceFromFloat(0)
		status = ExecutionStatusRejected
	}

	result := &ExecutionResult{
		OrderID:         order.OrderID,
		ExecutedQuantity: executedQuantity,
		ExecutedPrice:   executedPrice,
		ExecutionTime:   executionTime,
		VenueID:         1, // Default venue
		Status:          status,
	}

	processingTime := time.Since(start)
	ee.performanceCounters.UpdateOrderLatency(processingTime)

	return result
}

// GetExecutionResults returns execution results for an order
func (ee *ExecutionEngine) GetExecutionResults(orderID core.OrderID) []*ExecutionResult {
	ee.mutex.RLock()
	defer ee.mutex.RUnlock()

	results := make([]*ExecutionResult, 0)
	for _, result := range ee.executionResults {
		if result.OrderID == orderID {
			results = append(results, result)
		}
	}

	return results
}

// GetAllExecutionResults returns all execution results
func (ee *ExecutionEngine) GetAllExecutionResults() []*ExecutionResult {
	ee.mutex.RLock()
	defer ee.mutex.RUnlock()

	results := make([]*ExecutionResult, len(ee.executionResults))
	copy(results, ee.executionResults)

	return results
}

// GetQueueSize returns current queue size
func (ee *ExecutionEngine) GetQueueSize() int {
	ee.mutex.RLock()
	defer ee.mutex.RUnlock()

	return len(ee.executionQueue)
}

// IsQueueFull checks if execution queue is full
func (ee *ExecutionEngine) IsQueueFull() bool {
	ee.mutex.RLock()
	defer ee.mutex.RUnlock()

	return len(ee.executionQueue) >= ee.maxQueueSize
}

// ClearResults clears all execution results
func (ee *ExecutionEngine) ClearResults() {
	ee.mutex.Lock()
	defer ee.mutex.Unlock()

	ee.executionResults = ee.executionResults[:0]
}

// GetPerformanceCounters returns performance statistics
func (ee *ExecutionEngine) GetPerformanceCounters() *core.PerformanceCounters {
	return ee.performanceCounters
}

// ResetPerformanceCounters resets performance counters
func (ee *ExecutionEngine) ResetPerformanceCounters() {
	ee.performanceCounters = &core.PerformanceCounters{}
}

// SimulateMarketExecution simulates realistic market execution
func (ee *ExecutionEngine) SimulateMarketExecution(order *core.Order, marketPrice core.Price, slippage float64) *ExecutionResult {
	// Apply slippage to simulate realistic execution
	slippageFactor := 1.0 + slippage
	if order.Side == core.OrderSideSell {
		slippageFactor = 1.0 - slippage
	}

	executedPrice := marketPrice.Mul(core.DecimalFromFloat(slippageFactor))

	// Simulate partial fills for large orders
	executedQuantity := order.Quantity
	if order.Quantity.GreaterThan(core.QuantityFromFloat(1000)) {
		// Large orders might get partial fills
		fillRatio := 0.8 // 80% fill
		executedQuantity = order.Quantity.Mul(core.DecimalFromFloat(fillRatio))
	}

	return &ExecutionResult{
		OrderID:         order.OrderID,
		ExecutedQuantity: executedQuantity,
		ExecutedPrice:   executedPrice,
		ExecutionTime:   core.CurrentTimestamp(),
		VenueID:         1,
		Status:          ExecutionStatusFilled,
	}
}

// GetExecutionStats returns execution statistics
func (ee *ExecutionEngine) GetExecutionStats() map[string]interface{} {
	ee.mutex.RLock()
	defer ee.mutex.RUnlock()

	stats := make(map[string]interface{})
	stats["queue_size"] = len(ee.executionQueue)
	stats["total_executions"] = len(ee.executionResults)
	stats["queue_capacity"] = ee.maxQueueSize

	// Calculate fill rates
	totalRequested := core.QuantityFromFloat(0)
	totalExecuted := core.QuantityFromFloat(0)

	for _, result := range ee.executionResults {
		if result.Status == ExecutionStatusFilled || result.Status == ExecutionStatusPartial {
			totalExecuted = totalExecuted.Add(result.ExecutedQuantity)
		}
	}

	if totalRequested.GreaterThan(core.QuantityFromFloat(0)) {
		fillRate, _ := totalExecuted.Div(totalRequested).Float64()
		stats["fill_rate"] = fillRate
	} else {
		stats["fill_rate"] = 0.0
	}

	return stats
}
