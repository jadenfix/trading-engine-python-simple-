package cache

import (
	"sync"
	"time"

	"go-trading-engine/pkg/types"
)

// LruCache provides a high-performance LRU cache
type LruCache[K comparable, V any] struct {
	capacity       int
	cache          map[K]*cacheItem[K, V]
	accessCounter  uint64
	head, tail     *cacheItem[K, V]
	mutex          sync.RWMutex
}

type cacheItem[K comparable, V any] struct {
	key          K
	value        V
	accessCount  uint64
	prev, next   *cacheItem[K, V]
}

// NewLruCache creates a new LRU cache
func NewLruCache[K comparable, V any](capacity int) *LruCache[K, V] {
	return &LruCache[K, V]{
		capacity: capacity,
		cache:    make(map[K]*cacheItem[K, V], capacity),
	}
}

// Get retrieves a value from the cache
func (c *LruCache[K, V]) Get(key K) (V, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.cache[key]; exists {
		c.accessCounter++
		item.accessCount = c.accessCounter
		c.moveToFront(item)
		return item.value, true
	}

	var zero V
	return zero, false
}

// Put adds or updates a value in the cache
func (c *LruCache[K, V]) Put(key K, value V) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.accessCounter++

	if item, exists := c.cache[key]; exists {
		item.value = value
		item.accessCount = c.accessCounter
		c.moveToFront(item)
		return
	}

	// Create new item
	item := &cacheItem[K, V]{
		key:         key,
		value:       value,
		accessCount: c.accessCounter,
	}

	c.cache[key] = item
	c.addToFront(item)

	// Evict if over capacity
	if len(c.cache) > c.capacity {
		c.removeLRU()
	}
}

// Contains checks if key exists in cache
func (c *LruCache[K, V]) Contains(key K) bool {
	c.mutex.RLock()
	_, exists := c.cache[key]
	c.mutex.RUnlock()
	return exists
}

// Len returns current cache size
func (c *LruCache[K, V]) Len() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return len(c.cache)
}

// Clear removes all items from cache
func (c *LruCache[K, V]) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cache = make(map[K]*cacheItem[K, V], c.capacity)
	c.head = nil
	c.tail = nil
	c.accessCounter = 0
}

// moveToFront moves an item to the front of the LRU list
func (c *LruCache[K, V]) moveToFront(item *cacheItem[K, V]) {
	if c.head == item {
		return
	}

	c.removeFromList(item)
	c.addToFront(item)
}

// addToFront adds an item to the front of the LRU list
func (c *LruCache[K, V]) addToFront(item *cacheItem[K, V]) {
	item.prev = nil
	item.next = c.head

	if c.head != nil {
		c.head.prev = item
	}
	c.head = item

	if c.tail == nil {
		c.tail = item
	}
}

// removeFromList removes an item from the LRU list
func (c *LruCache[K, V]) removeFromList(item *cacheItem[K, V]) {
	if item.prev != nil {
		item.prev.next = item.next
	} else {
		c.head = item.next
	}

	if item.next != nil {
		item.next.prev = item.prev
	} else {
		c.tail = item.prev
	}

	item.prev = nil
	item.next = nil
}

// removeLRU removes the least recently used item
func (c *LruCache[K, V]) removeLRU() {
	if c.tail == nil {
		return
	}

	delete(c.cache, c.tail.key)
	c.removeFromList(c.tail)
}

// SignalCache provides thread-safe signal caching
type SignalCache struct {
	cache *LruCache[signalCacheKey, []types.Signal]
}

type signalCacheKey struct {
	symbolID   types.SymbolID
	strategyID uint32
}

// NewSignalCache creates a new signal cache
func NewSignalCache(capacity int) *SignalCache {
	return &SignalCache{
		cache: NewLruCache[signalCacheKey, []types.Signal](capacity),
	}
}

// GetSignals retrieves cached signals
func (sc *SignalCache) GetSignals(symbolID types.SymbolID, strategyID uint32) ([]types.Signal, bool) {
	key := signalCacheKey{symbolID: symbolID, strategyID: strategyID}
	return sc.cache.Get(key)
}

// CacheSignals stores signals in cache
func (sc *SignalCache) CacheSignals(symbolID types.SymbolID, strategyID uint32, signals []types.Signal) {
	key := signalCacheKey{symbolID: symbolID, strategyID: strategyID}
	sc.cache.Put(key, signals)
}

// Clear removes all cached signals
func (sc *SignalCache) Clear() {
	sc.cache.Clear()
}

// Stats returns cache statistics
func (sc *SignalCache) Stats() (size int, capacity int) {
	return sc.cache.Len(), sc.cache.capacity
}

// PriceHistoryCache provides thread-safe price history caching
type PriceHistoryCache struct {
	cache    map[types.SymbolID][]types.Price
	maxHistory int
	mutex    sync.RWMutex
}

// NewPriceHistoryCache creates a new price history cache
func NewPriceHistoryCache(maxHistory int) *PriceHistoryCache {
	return &PriceHistoryCache{
		cache:      make(map[types.SymbolID][]types.Price),
		maxHistory: maxHistory,
	}
}

// AddPrice adds a price to history
func (phc *PriceHistoryCache) AddPrice(symbolID types.SymbolID, price types.Price) {
	phc.mutex.Lock()
	defer phc.mutex.Unlock()

	history := phc.cache[symbolID]
	history = append(history, price)

	// Keep only recent history
	if len(history) > phc.maxHistory {
		history = history[len(history)-phc.maxHistory:]
	}

	phc.cache[symbolID] = history
}

// GetHistory retrieves price history
func (phc *PriceHistoryCache) GetHistory(symbolID types.SymbolID) []types.Price {
	phc.mutex.RLock()
	defer phc.mutex.RUnlock()

	if history, exists := phc.cache[symbolID]; exists {
		// Return a copy to prevent external modification
		result := make([]types.Price, len(history))
		copy(result, history)
		return result
	}
	return nil
}

// GetRecentPrices retrieves recent prices
func (phc *PriceHistoryCache) GetRecentPrices(symbolID types.SymbolID, n int) []types.Price {
	phc.mutex.RLock()
	defer phc.mutex.RUnlock()

	if history, exists := phc.cache[symbolID]; exists {
		start := len(history) - n
		if start < 0 {
			start = 0
		}
		result := make([]types.Price, len(history)-start)
		copy(result, history[start:])
		return result
	}
	return nil
}

// ClearSymbol clears history for a symbol
func (phc *PriceHistoryCache) ClearSymbol(symbolID types.SymbolID) {
	phc.mutex.Lock()
	defer phc.mutex.Unlock()
	delete(phc.cache, symbolID)
}

// ClearAll clears all histories
func (phc *PriceHistoryCache) ClearAll() {
	phc.mutex.Lock()
	defer phc.mutex.Unlock()
	phc.cache = make(map[types.SymbolID][]types.Price)
}

// IndicatorCache provides computed indicator caching
type IndicatorCache struct {
	cache  map[indicatorCacheKey]indicatorCacheValue
	ttl    time.Duration
	mutex  sync.RWMutex
}

type indicatorCacheKey struct {
	symbolID   types.SymbolID
	indicatorName string
}

type indicatorCacheValue struct {
	values    []float64
	timestamp time.Time
}

// NewIndicatorCache creates a new indicator cache
func NewIndicatorCache(ttl time.Duration) *IndicatorCache {
	return &IndicatorCache{
		cache: make(map[indicatorCacheKey]indicatorCacheValue),
		ttl:   ttl,
	}
}

// GetIndicator retrieves cached indicator values
func (ic *IndicatorCache) GetIndicator(symbolID types.SymbolID, indicatorName string) ([]float64, bool) {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()

	key := indicatorCacheKey{symbolID: symbolID, indicatorName: indicatorName}
	if value, exists := ic.cache[key]; exists {
		if time.Since(value.timestamp) < ic.ttl {
			// Return a copy to prevent external modification
			result := make([]float64, len(value.values))
			copy(result, value.values)
			return result, true
		}
		// Expired, remove it
		delete(ic.cache, key)
	}

	return nil, false
}

// CacheIndicator stores indicator values
func (ic *IndicatorCache) CacheIndicator(symbolID types.SymbolID, indicatorName string, values []float64) {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()

	key := indicatorCacheKey{symbolID: symbolID, indicatorName: indicatorName}

	// Copy values to prevent external modification
	cachedValues := make([]float64, len(values))
	copy(cachedValues, values)

	ic.cache[key] = indicatorCacheValue{
		values:    cachedValues,
		timestamp: time.Now(),
	}

	// Clean expired entries occasionally
	if len(ic.cache)%100 == 0 {
		ic.cleanExpired()
	}
}

// cleanExpired removes expired cache entries
func (ic *IndicatorCache) cleanExpired() {
	now := time.Now()
	for key, value := range ic.cache {
		if now.Sub(value.timestamp) >= ic.ttl {
			delete(ic.cache, key)
		}
	}
}

// Clear removes all cached indicators
func (ic *IndicatorCache) Clear() {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	ic.cache = make(map[indicatorCacheKey]indicatorCacheValue)
}

// Stats returns cache statistics
func (ic *IndicatorCache) Stats() int {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()
	return len(ic.cache)
}

// MultiLevelCache provides comprehensive caching system
type MultiLevelCache struct {
	signalCache     *SignalCache
	priceHistory    *PriceHistoryCache
	indicators      *IndicatorCache
}

// NewMultiLevelCache creates a new multi-level cache system
func NewMultiLevelCache() *MultiLevelCache {
	return &MultiLevelCache{
		signalCache:  NewSignalCache(1000),
		priceHistory: NewPriceHistoryCache(1000),
		indicators:   NewIndicatorCache(5 * time.Minute),
	}
}

// PreloadSymbol preloads data for a symbol to warm up caches
func (mlc *MultiLevelCache) PreloadSymbol(symbolID types.SymbolID, recentPrices []types.Price) {
	for _, price := range recentPrices {
		mlc.priceHistory.AddPrice(symbolID, price)
	}
}

// Stats returns comprehensive cache statistics
func (mlc *MultiLevelCache) Stats() CacheStats {
	signalSize, signalCapacity := mlc.signalCache.Stats()

	return CacheStats{
		SignalCacheSize:     signalSize,
		SignalCacheCapacity: signalCapacity,
		PriceHistorySymbols: len(mlc.priceHistory.cache),
		IndicatorCacheSize:  mlc.indicators.Stats(),
	}
}

// ClearAll clears all caches
func (mlc *MultiLevelCache) ClearAll() {
	mlc.signalCache.Clear()
	mlc.priceHistory.ClearAll()
	mlc.indicators.Clear()
}

// CacheStats represents cache statistics
type CacheStats struct {
	SignalCacheSize     int `json:"signal_cache_size"`
	SignalCacheCapacity int `json:"signal_cache_capacity"`
	PriceHistorySymbols int `json:"price_history_symbols"`
	IndicatorCacheSize  int `json:"indicator_cache_size"`
}
