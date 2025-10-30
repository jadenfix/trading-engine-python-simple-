use crate::core::types::*;
use std::collections::HashMap;
use std::sync::RwLock;

/// High-performance LRU cache with fixed capacity
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, usize)>, // (value, access_count)
    access_counter: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create new LRU cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            access_counter: 0,
        }
    }

    /// Get value from cache, updating access pattern
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some((value, count)) = self.map.get_mut(key) {
            *count = self.access_counter;
            self.access_counter += 1;
            Some(value)
        } else {
            None
        }
    }

    /// Insert value into cache, evicting least recently used if necessary
    pub fn insert(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity {
            // Find least recently used item
            let mut min_count = usize::MAX;
            let mut lru_key = None;

            for (k, (_, count)) in &self.map {
                if *count < min_count {
                    min_count = *count;
                    lru_key = Some(k.clone());
                }
            }

            if let Some(lru_key) = lru_key {
                self.map.remove(&lru_key);
            }
        }

        self.map.insert(key, (value, self.access_counter));
        self.access_counter += 1;
    }

    /// Check if key exists in cache
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.map.clear();
        self.access_counter = 0;
    }
}

/// Thread-safe signal cache for high-frequency trading
pub struct SignalCache {
    cache: RwLock<LruCache<(SymbolId, u32), Vec<Signal>>>, // (symbol_id, strategy_id) -> signals
}

impl SignalCache {
    /// Create new signal cache with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(capacity)),
        }
    }

    /// Get cached signals for symbol and strategy
    pub fn get_signals(&self, symbol_id: SymbolId, strategy_id: u32) -> Option<Vec<Signal>> {
        let mut cache = self.cache.write().unwrap();
        cache.get(&(symbol_id, strategy_id)).cloned()
    }

    /// Cache signals for symbol and strategy
    pub fn cache_signals(&self, symbol_id: SymbolId, strategy_id: u32, signals: Vec<Signal>) {
        let mut cache = self.cache.write().unwrap();
        cache.insert((symbol_id, strategy_id), signals);
    }

    /// Clear all cached signals
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize) {
        let cache = self.cache.read().unwrap();
        (cache.len(), cache.capacity)
    }
}

impl Default for SignalCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Thread-safe price history cache for technical indicators
pub struct PriceHistoryCache {
    cache: RwLock<HashMap<SymbolId, Vec<Price>>>,
    max_history: usize,
}

impl PriceHistoryCache {
    /// Create new price history cache
    pub fn new(max_history: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_history,
        }
    }

    /// Add price to history for symbol
    pub fn add_price(&self, symbol_id: SymbolId, price: Price) {
        let mut cache = self.cache.write().unwrap();

        let history = cache.entry(symbol_id).or_insert_with(Vec::new);
        history.push(price);

        // Keep only recent history
        if history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Get price history for symbol
    pub fn get_history(&self, symbol_id: SymbolId) -> Option<Vec<Price>> {
        let cache = self.cache.read().unwrap();
        cache.get(&symbol_id).cloned()
    }

    /// Get recent prices (last n prices)
    pub fn get_recent_prices(&self, symbol_id: SymbolId, n: usize) -> Option<Vec<Price>> {
        let cache = self.cache.read().unwrap();
        cache.get(&symbol_id)
            .and_then(|history| {
                let start = history.len().saturating_sub(n);
                Some(history[start..].to_vec())
            })
    }

    /// Clear history for symbol
    pub fn clear_symbol(&self, symbol_id: SymbolId) {
        let mut cache = self.cache.write().unwrap();
        cache.remove(&symbol_id);
    }

    /// Clear all histories
    pub fn clear_all(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

impl Default for PriceHistoryCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Thread-safe computed indicator cache
pub struct IndicatorCache {
    cache: RwLock<HashMap<(SymbolId, String), (Vec<f64>, Timestamp)>>, // (symbol, indicator_name) -> (values, timestamp)
    ttl_ns: Timestamp, // Time to live in nanoseconds
}

impl IndicatorCache {
    /// Create new indicator cache with TTL
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            ttl_ns: ttl_seconds * 1_000_000_000,
        }
    }

    /// Get cached indicator values
    pub fn get_indicator(&self, symbol_id: SymbolId, indicator_name: &str) -> Option<Vec<f64>> {
        let cache = self.cache.read().unwrap();
        let key = (symbol_id, indicator_name.to_string());

        if let Some((values, timestamp)) = cache.get(&key) {
            if current_timestamp() - timestamp < self.ttl_ns {
                return Some(values.clone());
            }
        }
        None
    }

    /// Cache indicator values
    pub fn cache_indicator(&self, symbol_id: SymbolId, indicator_name: String, values: Vec<f64>) {
        let mut cache = self.cache.write().unwrap();
        let key = (symbol_id, indicator_name);
        let timestamp = current_timestamp();

        // Clean expired entries occasionally
        if cache.len() % 100 == 0 {
            self.clean_expired(&mut cache);
        }

        cache.insert(key, (values, timestamp));
    }

    /// Clean expired cache entries
    fn clean_expired(&self, cache: &mut HashMap<(SymbolId, String), (Vec<f64>, Timestamp)>) {
        let current_time = current_timestamp();
        cache.retain(|_, (_, timestamp)| current_time - timestamp < self.ttl_ns);
    }

    /// Clear all cached indicators
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }
}

impl Default for IndicatorCache {
    fn default() -> Self {
        Self::new(300) // 5 minutes TTL
    }
}

/// Multi-level cache system for trading engine
pub struct MultiLevelCache {
    pub signal_cache: SignalCache,
    pub price_history: PriceHistoryCache,
    pub indicators: IndicatorCache,
}

impl MultiLevelCache {
    /// Create new multi-level cache system
    pub fn new() -> Self {
        Self {
            signal_cache: SignalCache::default(),
            price_history: PriceHistoryCache::default(),
            indicators: IndicatorCache::default(),
        }
    }

    /// Preload data for symbol to warm up caches
    pub fn preload_symbol(&self, symbol_id: SymbolId, recent_prices: &[Price]) {
        // Add recent prices to history
        for &price in recent_prices {
            self.price_history.add_price(symbol_id, price);
        }
    }

    /// Get comprehensive cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            signal_cache_size: self.signal_cache.stats().0,
            price_history_symbols: {
                let cache = self.price_history.cache.read().unwrap();
                cache.len()
            },
            indicator_cache_size: self.indicators.stats(),
        }
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.signal_cache.clear();
        self.price_history.clear_all();
        self.indicators.clear();
    }
}

/// Cache statistics structure
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub signal_cache_size: usize,
    pub price_history_symbols: usize,
    pub indicator_cache_size: usize,
}

impl Default for MultiLevelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;

    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::<i32, String>::new(3);

        // Insert items
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));

        // Access item to update LRU
        cache.get(&1);
        cache.get(&2);

        // Insert fourth item, should evict least recently used (3)
        cache.insert(4, "four".to_string());

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&4));
        assert!(!cache.contains_key(&3));
    }

    #[test]
    fn test_price_history_cache() {
        let cache = PriceHistoryCache::new(5);

        // Add prices
        cache.add_price(1, price_from_float(100.0));
        cache.add_price(1, price_from_float(101.0));
        cache.add_price(1, price_from_float(102.0));

        // Get history
        let history = cache.get_history(1).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(price_to_float(history[0]), 100.0);
        assert_eq!(price_to_float(history[1]), 101.0);
        assert_eq!(price_to_float(history[2]), 102.0);

        // Get recent prices
        let recent = cache.get_recent_prices(1, 2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(price_to_float(recent[0]), 101.0);
        assert_eq!(price_to_float(recent[1]), 102.0);

        // Add more prices to test capacity limit
        for i in 3..8 {
            cache.add_price(1, price_from_float(100.0 + i as f64));
        }

        let history = cache.get_history(1).unwrap();
        assert_eq!(history.len(), 5); // Should be capped at 5
    }

    #[test]
    fn test_indicator_cache() {
        let cache = IndicatorCache::new(1); // 1 second TTL

        // Cache indicator
        let values = vec![10.0, 20.0, 30.0];
        cache.cache_indicator(1, "sma".to_string(), values.clone());

        // Get cached indicator
        let cached = cache.get_indicator(1, "sma").unwrap();
        assert_eq!(cached, values);

        // Test cache stats
        assert_eq!(cache.stats(), 1);
    }

    #[test]
    fn test_signal_cache() {
        let cache = SignalCache::new(10);

        // Create test signals
        let signals = vec![
            Signal {
                symbol_id: 1,
                signal: SignalType::Long,
                confidence: 0.8,
                timestamp: current_timestamp(),
                strategy_id: 1,
                target_price: price_from_float(100.0),
                target_quantity: Decimal::from(100),
            }
        ];

        // Cache signals
        cache.cache_signals(1, 1, signals.clone());

        // Get cached signals
        let cached = cache.get_signals(1, 1).unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].symbol_id, 1);
        assert_eq!(cached[0].signal, SignalType::Long);

        // Test stats
        let (size, capacity) = cache.stats();
        assert_eq!(size, 1);
        assert_eq!(capacity, 10);
    }
}
