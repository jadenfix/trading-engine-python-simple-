use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Thread-local memory pool for high-performance allocation
pub struct MemoryPool {
    block_size: usize,
    block_count: usize,
    free_list: UnsafeCell<Vec<NonNull<u8>>>,
    allocations: UnsafeCell<usize>,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(block_size: usize, block_count: usize) -> Self {
        let mut free_list = Vec::with_capacity(block_count);

        // Pre-allocate blocks
        for _ in 0..block_count {
            let layout = Layout::array::<u8>(block_size).unwrap();
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                panic!("Failed to allocate memory block");
            }
            free_list.push(unsafe { NonNull::new_unchecked(ptr) });
        }

        Self {
            block_size,
            block_count,
            free_list: UnsafeCell::new(free_list),
            allocations: UnsafeCell::new(0),
        }
    }

    /// Allocate memory from pool
    pub fn allocate(&self) -> Option<NonNull<u8>> {
        let free_list = unsafe { &mut *self.free_list.get() };
        let allocations = unsafe { &mut *self.allocations.get() };

        if let Some(ptr) = free_list.pop() {
            *allocations += 1;
            Some(ptr)
        } else {
            None // Pool exhausted
        }
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let free_list = unsafe { &mut *self.free_list.get() };
        let allocations = unsafe { &mut *self.allocations.get() };

        if *allocations > 0 {
            free_list.push(ptr);
            *allocations -= 1;
        }
    }

    /// Get allocation statistics
    pub fn stats(&self) -> (usize, usize) {
        let free_list = unsafe { &*self.free_list.get() };
        let allocations = unsafe { &*self.allocations.get() };
        (*allocations, free_list.len())
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let free_list = unsafe { &mut *self.free_list.get() };

        // Deallocate all remaining blocks
        for ptr in free_list.drain(..) {
            let layout = Layout::array::<u8>(self.block_size).unwrap();
            unsafe { dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// Lock-free queue for high-performance message passing
pub struct LockFreeQueue<T> {
    head: AtomicUsize,
    tail: AtomicUsize,
    buffer: Vec<UnsafeCell<Option<T>>>,
    capacity: usize,
}

impl<T> LockFreeQueue<T> {
    /// Create new lock-free queue with given capacity
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(None));
        }

        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            buffer,
            capacity,
        }
    }

    /// Try to push item to queue (non-blocking)
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) % self.capacity;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(item); // Queue full
        }

        unsafe {
            *self.buffer[tail].get() = Some(item);
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Try to pop item from queue (non-blocking)
    pub fn try_pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);

        if head == self.tail.load(Ordering::Acquire) {
            return None; // Queue empty
        }

        let item = unsafe {
            (*self.buffer[head].get()).take()
        };

        let next_head = (head + 1) % self.capacity;
        self.head.store(next_head, Ordering::Release);

        item
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Acquire)
    }

    /// Check if queue is full
    pub fn is_full(&self) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) % self.capacity;
        next_tail == self.head.load(Ordering::Acquire)
    }

    /// Get current queue length
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if tail >= head {
            tail - head
        } else {
            self.capacity - head + tail
        }
    }
}

/// Cache-aligned allocator for performance-critical structures
pub struct CacheAlignedAllocator;

impl CacheAlignedAllocator {
    /// Allocate cache-aligned memory
    pub fn allocate<T>(count: usize) -> Option<NonNull<T>> {
        let layout = Layout::array::<T>(count)
            .ok()?
            .align_to(crate::core::types::CACHE_LINE_SIZE)
            .ok()?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { NonNull::new_unchecked(ptr as *mut T) })
        }
    }

    /// Deallocate cache-aligned memory
    pub fn deallocate<T>(ptr: NonNull<T>, count: usize) {
        let layout = Layout::array::<T>(count)
            .unwrap()
            .align_to(crate::core::types::CACHE_LINE_SIZE)
            .unwrap();

        unsafe { dealloc(ptr.as_ptr() as *mut u8, layout) };
    }
}

/// Thread-local storage for performance-critical data
pub struct ThreadLocalStorage<T: Clone + Default> {
    data: std::thread::LocalKey<UnsafeCell<T>>,
}

impl<T: Clone + Default> ThreadLocalStorage<T> {
    /// Create new thread-local storage
    pub fn new() -> Self {
        Self {
            data: std::thread::LocalKey::new(|| UnsafeCell::new(T::default())),
        }
    }

    /// Get reference to thread-local data
    pub fn get(&'static self) -> &'static mut T {
        self.data.with(|cell| unsafe { &mut *cell.get() })
    }
}

/// Performance-optimized vector that uses cache-aligned allocation
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> AlignedVec<T> {
    /// Create new aligned vector with capacity
    pub fn with_capacity(capacity: usize) -> Option<Self> {
        let ptr = CacheAlignedAllocator::allocate::<T>(capacity)?;
        Some(Self {
            ptr,
            len: 0,
            capacity,
        })
    }

    /// Push item to vector
    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.len >= self.capacity {
            return Err(item);
        }

        unsafe {
            std::ptr::write(self.ptr.as_ptr().add(self.len), item);
        }
        self.len += 1;
        Ok(())
    }

    /// Get item at index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            None
        } else {
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        }
    }

    /// Get mutable reference to item at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            None
        } else {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        }
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear vector (doesn't deallocate)
    pub fn clear(&mut self) {
        // Drop elements if they implement Drop
        for i in 0..self.len {
            unsafe {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        self.len = 0;
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }

        // Deallocate memory
        CacheAlignedAllocator::deallocate(self.ptr, self.capacity);
    }
}

/// Memory arena for bulk allocations
pub struct MemoryArena {
    buffer: Vec<u8>,
    offset: AtomicUsize,
}

impl MemoryArena {
    /// Create new memory arena with given size
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![0; size],
            offset: AtomicUsize::new(0),
        }
    }

    /// Allocate from arena
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let current_offset = self.offset.load(Ordering::Relaxed);
        let aligned_offset = (current_offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.buffer.len() {
            return None; // Out of memory
        }

        self.offset.store(aligned_offset + size, Ordering::Relaxed);

        let ptr = unsafe { self.buffer.as_ptr().add(aligned_offset) };
        Some(unsafe { NonNull::new_unchecked(ptr as *mut u8) })
    }

    /// Reset arena for reuse
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    /// Get memory usage
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(64, 10);
        let (allocations, free) = pool.stats();
        assert_eq!(allocations, 0);
        assert_eq!(free, 10);

        // Allocate some blocks
        let block1 = pool.allocate().unwrap();
        let block2 = pool.allocate().unwrap();

        let (allocations, free) = pool.stats();
        assert_eq!(allocations, 2);
        assert_eq!(free, 8);

        // Deallocate
        pool.deallocate(block1);
        pool.deallocate(block2);

        let (allocations, free) = pool.stats();
        assert_eq!(allocations, 0);
        assert_eq!(free, 10);
    }

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::<i32>::new(4);

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        // Push items
        assert!(queue.try_push(1).is_ok());
        assert!(queue.try_push(2).is_ok());
        assert!(queue.try_push(3).is_ok());

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());

        // Pop items
        assert_eq!(queue.try_pop(), Some(1));
        assert_eq!(queue.try_pop(), Some(2));
        assert_eq!(queue.try_pop(), Some(3));
        assert_eq!(queue.try_pop(), None);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_memory_arena() {
        let arena = MemoryArena::new(1024);

        // Allocate some memory
        let ptr1 = arena.allocate(64, 8).unwrap();
        let ptr2 = arena.allocate(128, 16).unwrap();

        assert_eq!(arena.used(), 192); // 64 + 128 with alignment

        // Reset arena
        arena.reset();
        assert_eq!(arena.used(), 0);

        // Allocate again
        let ptr3 = arena.allocate(256, 32).unwrap();
        assert_eq!(arena.used(), 256);
    }
}
