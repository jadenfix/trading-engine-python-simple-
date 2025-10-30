package order_engine

import (
	"go-trading-engine/internal/core"
	"go-trading-engine/pkg/types"
	"sync"
	"time"
)

// OrderEngine manages order lifecycle and matching
type OrderEngine struct {
	activeOrders       map[core.OrderID]*core.Order
	orderBook          *OrderBook
	nextOrderID        core.OrderID
	performanceCounters *core.PerformanceCounters
	mutex              sync.RWMutex
}

// NewOrderEngine creates a new order engine
func NewOrderEngine() *OrderEngine {
	return &OrderEngine{
		activeOrders:       make(map[core.OrderID]*core.Order),
		orderBook:          NewOrderBook(),
		nextOrderID:        1,
		performanceCounters: &core.PerformanceCounters{},
	}
}

// SubmitOrder submits a new order
func (oe *OrderEngine) SubmitOrder(order *core.Order) core.OrderID {
	start := time.Now()

	oe.mutex.Lock()
	defer oe.mutex.Unlock()

	// Assign order ID
	order.OrderID = oe.nextOrderID
	oe.nextOrderID++

	// Set timestamps
	currentTime := core.CurrentTimestamp()
	order.CreateTime = currentTime
	order.UpdateTime = currentTime

	// Add to active orders
	oe.activeOrders[order.OrderID] = order

	// Add to order book if limit order
	if order.OrderType == core.OrderTypeLimit {
		oe.orderBook.AddLimitOrder(order)
	}

	processingTime := time.Since(start)
	oe.performanceCounters.UpdateOrderLatency(processingTime)
	oe.performanceCounters.OrdersSubmitted++

	return order.OrderID
}

// CancelOrder cancels an existing order
func (oe *OrderEngine) CancelOrder(orderID core.OrderID) bool {
	oe.mutex.Lock()
	defer oe.mutex.Unlock()

	if order, exists := oe.activeOrders[orderID]; exists {
		if order.Status == core.OrderStatusPending {
			order.Status = core.OrderStatusCancelled
			order.UpdateTime = core.CurrentTimestamp()
			delete(oe.activeOrders, orderID)
			oe.orderBook.RemoveOrder(orderID)
			return true
		}
	}

	return false
}

// ModifyOrder modifies an existing order
func (oe *OrderEngine) ModifyOrder(orderID core.OrderID, newPrice core.Price, newQuantity core.Quantity) bool {
	oe.mutex.Lock()
	defer oe.mutex.Unlock()

	if order, exists := oe.activeOrders[orderID]; exists {
		if order.Status == core.OrderStatusPending {
			// Remove from old position in order book
			oe.orderBook.RemoveOrder(orderID)

			// Update order
			order.Price = newPrice
			order.Quantity = newQuantity
			order.UpdateTime = core.CurrentTimestamp()

			// Re-add to order book if still limit order
			if order.OrderType == core.OrderTypeLimit {
				oe.orderBook.AddLimitOrder(order)
			}

			return true
		}
	}

	return false
}

// GetOrder retrieves an order by ID
func (oe *OrderEngine) GetOrder(orderID core.OrderID) (*core.Order, bool) {
	oe.mutex.RLock()
	defer oe.mutex.RUnlock()

	order, exists := oe.activeOrders[orderID]
	return order, exists
}

// GetActiveOrders returns all active orders
func (oe *OrderEngine) GetActiveOrders() []*core.Order {
	oe.mutex.RLock()
	defer oe.mutex.RUnlock()

	orders := make([]*core.Order, 0, len(oe.activeOrders))
	for _, order := range oe.activeOrders {
		orders = append(orders, order)
	}

	return orders
}

// GetOrderBookSnapshot returns current order book snapshot
func (oe *OrderEngine) GetOrderBookSnapshot(symbolID core.SymbolID) *OrderBookSnapshot {
	return oe.orderBook.GetSnapshot(symbolID)
}

// MatchOrders attempts to match orders (simplified implementation)
func (oe *OrderEngine) MatchOrders(symbolID core.SymbolID) []OrderMatch {
	return oe.orderBook.MatchOrders(symbolID)
}

// GetPerformanceCounters returns performance statistics
func (oe *OrderEngine) GetPerformanceCounters() *core.PerformanceCounters {
	return oe.performanceCounters
}

// ResetPerformanceCounters resets performance counters
func (oe *OrderEngine) ResetPerformanceCounters() {
	oe.performanceCounters = &core.PerformanceCounters{}
}

// OrderBook represents a limit order book
type OrderBook struct {
	bids map[core.Price][]*core.Order // Price -> Orders (sorted descending)
	asks map[core.Price][]*core.Order // Price -> Orders (sorted ascending)
	mutex sync.RWMutex
}

// NewOrderBook creates a new order book
func NewOrderBook() *OrderBook {
	return &OrderBook{
		bids: make(map[core.Price][]*core.Order),
		asks: make(map[core.Price][]*core.Order),
	}
}

// AddLimitOrder adds a limit order to the book
func (ob *OrderBook) AddLimitOrder(order *core.Order) {
	ob.mutex.Lock()
	defer ob.mutex.Unlock()

	if order.OrderType != core.OrderTypeLimit {
		return
	}

	if order.Side == core.OrderSideBuy {
		ob.bids[order.Price] = append(ob.bids[order.Price], order)
	} else {
		ob.asks[order.Price] = append(ob.asks[order.Price], order)
	}
}

// RemoveOrder removes an order from the book
func (ob *OrderBook) RemoveOrder(orderID core.OrderID) {
	ob.mutex.Lock()
	defer ob.mutex.Unlock()

	// Remove from bids
	for price, orders := range ob.bids {
		for i, order := range orders {
			if order.OrderID == orderID {
				ob.bids[price] = append(orders[:i], orders[i+1:]...)
				if len(ob.bids[price]) == 0 {
					delete(ob.bids, price)
				}
				return
			}
		}
	}

	// Remove from asks
	for price, orders := range ob.asks {
		for i, order := range orders {
			if order.OrderID == orderID {
				ob.asks[price] = append(orders[:i], orders[i+1:]...)
				if len(ob.asks[price]) == 0 {
					delete(ob.asks, price)
				}
				return
			}
		}
	}
}

// GetSnapshot returns a snapshot of the order book
func (ob *OrderBook) GetSnapshot(symbolID core.SymbolID) *OrderBookSnapshot {
	ob.mutex.RLock()
	defer ob.mutex.RUnlock()

	snapshot := &OrderBookSnapshot{SymbolID: symbolID}

	// Find best bid
	for price := range ob.bids {
		if snapshot.BestBid.IsZero() || price.GreaterThan(snapshot.BestBid) {
			snapshot.BestBid = price
		}
	}

	// Find best ask
	for price := range ob.asks {
		if snapshot.BestAsk.IsZero() || price.LessThan(snapshot.BestAsk) {
			snapshot.BestAsk = price
		}
	}

	// Calculate spread
	if !snapshot.BestBid.IsZero() && !snapshot.BestAsk.IsZero() {
		spread, _ := snapshot.BestAsk.Sub(snapshot.BestBid).Float64()
		snapshot.Spread = spread
	}

	// Calculate volumes
	if bids, exists := ob.bids[snapshot.BestBid]; exists {
		totalVolume := core.QuantityFromFloat(0)
		for _, order := range bids {
			totalVolume = totalVolume.Add(order.Quantity)
		}
		snapshot.BidVolume = totalVolume
	}

	if asks, exists := ob.asks[snapshot.BestAsk]; exists {
		totalVolume := core.QuantityFromFloat(0)
		for _, order := range asks {
			totalVolume = totalVolume.Add(order.Quantity)
		}
		snapshot.AskVolume = totalVolume
	}

	snapshot.Timestamp = core.CurrentTimestamp()

	return snapshot
}

// MatchOrders attempts to match orders (simplified)
func (ob *OrderBook) MatchOrders(symbolID core.SymbolID) []OrderMatch {
	ob.mutex.Lock()
	defer ob.mutex.Unlock()

	var matches []OrderMatch

	// Find crossing orders (simplified matching)
	for bidPrice, bidOrders := range ob.bids {
		for askPrice, askOrders := range ob.asks {
			// Check if bid price >= ask price
			if bidPrice.GreaterThanOrEqual(askPrice) {
				// Match orders (simplified - just record potential matches)
				for _, bidOrder := range bidOrders {
					for _, askOrder := range askOrders {
						if bidOrder.SymbolID == symbolID && askOrder.SymbolID == symbolID {
							// Determine match quantity (min of both orders)
							matchQuantity := bidOrder.Quantity
							if askOrder.Quantity.LessThan(matchQuantity) {
								matchQuantity = askOrder.Quantity
							}

							matches = append(matches, OrderMatch{
								BidOrderID: bidOrder.OrderID,
								AskOrderID: askOrder.OrderID,
								Price:      askPrice, // Use ask price as match price
								Quantity:   matchQuantity,
								Timestamp:  core.CurrentTimestamp(),
							})

							// Update order quantities (simplified)
							bidOrder.Quantity = bidOrder.Quantity.Sub(matchQuantity)
							askOrder.Quantity = askOrder.Quantity.Sub(matchQuantity)

							// Remove filled orders
							if bidOrder.Quantity.IsZero() {
								ob.RemoveOrder(bidOrder.OrderID)
							}
							if askOrder.Quantity.IsZero() {
								ob.RemoveOrder(askOrder.OrderID)
							}
						}
					}
				}
			}
		}
	}

	return matches
}

// OrderBookSnapshot represents a snapshot of the order book
type OrderBookSnapshot struct {
	SymbolID  core.SymbolID `json:"symbol_id"`
	BestBid   core.Price    `json:"best_bid"`
	BestAsk   core.Price    `json:"best_ask"`
	BidVolume core.Quantity `json:"bid_volume"`
	AskVolume core.Quantity `json:"ask_volume"`
	Spread    float64       `json:"spread"`
	Timestamp core.Timestamp `json:"timestamp"`
}

// OrderMatch represents a matched order
type OrderMatch struct {
	BidOrderID core.OrderID `json:"bid_order_id"`
	AskOrderID core.OrderID `json:"ask_order_id"`
	Price      core.Price   `json:"price"`
	Quantity   core.Quantity `json:"quantity"`
	Timestamp  core.Timestamp `json:"timestamp"`
}
