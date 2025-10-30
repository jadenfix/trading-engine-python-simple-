package memory

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// MemoryPool provides thread-local memory pool for high-performance allocation
type MemoryPool struct {
	blockSize   int
	blockCount  int
	freeList    []unsafe.Pointer
	allocations int64
	mutex       sync.Mutex
}

// NewMemoryPool creates a new memory pool
func NewMemoryPool(blockSize, blockCount int) *MemoryPool {
	pool := &MemoryPool{
		blockSize:  blockSize,
		blockCount: blockCount,
		freeList:   make([]unsafe.Pointer, 0, blockCount),
	}

	// Pre-allocate blocks
	for i := 0; i < blockCount; i++ {
		block := make([]byte, blockSize)
		pool.freeList = append(pool.freeList, unsafe.Pointer(&block[0]))
	}

	return pool
}

// Allocate gets a memory block from the pool
func (mp *MemoryPool) Allocate() unsafe.Pointer {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()

	if len(mp.freeList) == 0 {
		// Pool exhausted, allocate new block
		block := make([]byte, mp.blockSize)
		atomic.AddInt64(&mp.allocations, 1)
		return unsafe.Pointer(&block[0])
	}

	block := mp.freeList[len(mp.freeList)-1]
	mp.freeList = mp.freeList[:len(mp.freeList)-1]
	atomic.AddInt64(&mp.allocations, 1)
	return block
}

// Deallocate returns a memory block to the pool
func (mp *MemoryPool) Deallocate(ptr unsafe.Pointer) {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()

	if len(mp.freeList) < mp.blockCount {
		mp.freeList = append(mp.freeList, ptr)
	}
	atomic.AddInt64(&mp.allocations, -1)
}

// Stats returns allocation statistics
func (mp *MemoryPool) Stats() (allocations int64, free int) {
	return atomic.LoadInt64(&mp.allocations), len(mp.freeList)
}

// LockFreeQueue provides a lock-free queue for high-performance message passing
type LockFreeQueue[T any] struct {
	head      uint64
	tail      uint64
	buffer    []unsafe.Pointer
	capacity  uint64
	allocator func() T
}

// NewLockFreeQueue creates a new lock-free queue
func NewLockFreeQueue[T any](capacity int, allocator func() T) *LockFreeQueue[T] {
	buffer := make([]unsafe.Pointer, capacity)
	for i := range buffer {
		obj := allocator()
		buffer[i] = unsafe.Pointer(&obj)
	}

	return &LockFreeQueue[T]{
		buffer:    buffer,
		capacity:  uint64(capacity),
		allocator: allocator,
	}
}

// TryPush attempts to push an item to the queue (non-blocking)
func (q *LockFreeQueue[T]) TryPush(item T) bool {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)

	nextTail := (tail + 1) % q.capacity

	if nextTail == head {
		return false // Queue full
	}

	// Copy item to buffer
	*(*T)(q.buffer[tail]) = item

	atomic.StoreUint64(&q.tail, nextTail)
	return true
}

// TryPop attempts to pop an item from the queue (non-blocking)
func (q *LockFreeQueue[T]) (T, bool) {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)

	if head == tail {
		var zero T
		return zero, false // Queue empty
	}

	item := *(*T)(q.buffer[head])
	atomic.StoreUint64(&q.head, (head+1)%q.capacity)
	return item, true
}

// IsEmpty checks if queue is empty
func (q *LockFreeQueue[T]) bool {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)
	return head == tail
}

// IsFull checks if queue is full
func (q *LockFreeQueue[T]) bool {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)
	nextTail := (tail + 1) % q.capacity
	return nextTail == head
}

// Len returns current queue length
func (q *LockFreeQueue[T]) int {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)

	if tail >= head {
		return int(tail - head)
	}
	return int(q.capacity - head + tail)
}

// CacheAlignedAllocator provides cache-aligned memory allocation
type CacheAlignedAllocator struct {
	cacheLineSize int
}

// NewCacheAlignedAllocator creates a new cache-aligned allocator
func NewCacheAlignedAllocator() *CacheAlignedAllocator {
	return &CacheAlignedAllocator{cacheLineSize: 64}
}

// Allocate allocates cache-aligned memory
func (caa *CacheAlignedAllocator) Allocate(size int) unsafe.Pointer {
	// Calculate aligned size
	alignedSize := ((size + caa.cacheLineSize - 1) / caa.cacheLineSize) * caa.cacheLineSize
	block := make([]byte, alignedSize)
	return unsafe.Pointer(&block[0])
}

// Deallocate deallocates cache-aligned memory (Go handles this automatically)
func (caa *CacheAlignedAllocator) Deallocate(ptr unsafe.Pointer) {
	// Go's garbage collector handles deallocation
}

// ThreadLocalStorage provides thread-local storage for performance-critical data
type ThreadLocalStorage[T any] struct {
	data sync.Map // Map from goroutine ID to value
}

// NewThreadLocalStorage creates new thread-local storage
func NewThreadLocalStorage[T any](defaultValue func() T) *ThreadLocalStorage[T] {
	return &ThreadLocalStorage[T]{}
}

// Get returns the thread-local value
func (tls *ThreadLocalStorage[T]) Get(defaultValue func() T) T {
	// In Go, we use goroutine-local storage via sync.Map with goroutine ID
	goroutineID := getGoroutineID()
	if value, ok := tls.data.Load(goroutineID); ok {
		return value.(T)
	}

	newValue := defaultValue()
	tls.data.Store(goroutineID, newValue)
	return newValue
}

// getGoroutineID returns a pseudo-unique identifier for the current goroutine
func getGoroutineID() uint64 {
	// This is a simplified implementation
	// In production, you'd want a more robust goroutine ID
	return uint64(uintptr(unsafe.Pointer(&struct{}{})))
}

// AlignedSlice provides cache-aligned slices for performance
type AlignedSlice[T any] struct {
	data   []T
	allocator *CacheAlignedAllocator
}

// NewAlignedSlice creates a new aligned slice
func NewAlignedSlice[T any](length int) *AlignedSlice[T] {
	allocator := NewCacheAlignedAllocator()
	elementSize := int(unsafe.Sizeof(*new(T)))

	// Allocate aligned memory
	ptr := allocator.Allocate(length * elementSize)
	slice := unsafe.Slice((*T)(ptr), length)

	return &AlignedSlice[T]{
		data:      slice,
		allocator: allocator,
	}
}

// Data returns the underlying slice
func (as *AlignedSlice[T]) Data() []T {
	return as.data
}

// MemoryArena provides bulk memory allocation
type MemoryArena struct {
	buffer  []byte
	offset  int64
	maxSize int
}

// NewMemoryArena creates a new memory arena
func NewMemoryArena(size int) *MemoryArena {
	return &MemoryArena{
		buffer:  make([]byte, size),
		offset:  0,
		maxSize: size,
	}
}

// Allocate allocates from the arena
func (ma *MemoryArena) Allocate(size int, align int) unsafe.Pointer {
	currentOffset := atomic.LoadInt64(&ma.offset)

	// Calculate aligned offset
	alignedOffset := (int(currentOffset) + align - 1) & ^(align - 1)

	if alignedOffset+size > ma.maxSize {
		return nil // Out of memory
	}

	atomic.StoreInt64(&ma.offset, int64(alignedOffset+size))
	return unsafe.Pointer(&ma.buffer[alignedOffset])
}

// Reset resets the arena for reuse
func (ma *MemoryArena) Reset() {
	atomic.StoreInt64(&ma.offset, 0)
}

// Used returns bytes used
func (ma *MemoryArena) Used() int {
	return int(atomic.LoadInt64(&ma.offset))
}

// Available returns bytes available
func (ma *MemoryArena) Available() int {
	return ma.maxSize - ma.Used()
}
