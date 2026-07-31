#ifndef YOLO_LOCK_FREE_QUEUE_HPP
#define YOLO_LOCK_FREE_QUEUE_HPP

#include <atomic>
#include <vector>
#include <optional>
#include <cstddef>
#include <cassert>

namespace yolo {


template<typename T>
class LockFreeQueue {
private:
    std::vector<T> buffer;
    std::atomic<size_t> head{0}; 
    std::atomic<size_t> tail{0}; 
    size_t buffer_capacity;

public:
    explicit LockFreeQueue(size_t max_size) 
        : buffer(max_size), buffer_capacity(max_size) {
            assert(max_size > 0);
    }

    bool enqueue(const T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % buffer_capacity;

        // Check if full
        if (next_tail == head.load(std::memory_order_acquire)) {
            size_t new_head = (head.load(std::memory_order_relaxed) + 1) % buffer_capacity;
            head.store(new_head, std::memory_order_release);
        }

        // Write data
        buffer[current_tail] = item;
        
        // Publish the new tail
        tail.store(next_tail, std::memory_order_release);
        
        return true;
    }

    // Enqueue with move semantics
    bool enqueue(T&& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % buffer_capacity;

        if (next_tail == head.load(std::memory_order_acquire)) {
            size_t new_head = (head.load(std::memory_order_relaxed) + 1) % buffer_capacity;
            head.store(new_head, std::memory_order_release);
        }

        buffer[current_tail] = std::move(item);
        tail.store(next_tail, std::memory_order_release);

        return true;
    }

    // Dequeue
    std::optional<T> dequeue() {
        size_t current_head = head.load(std::memory_order_relaxed);
        
        // Check if empty
        if (current_head == tail.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T item = std::move(buffer[current_head]);
        head.store((current_head + 1) % buffer_capacity, std::memory_order_release);
        return item;
    }

    bool isEmpty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t h = head.load(std::memory_order_acquire);
        size_t t = tail.load(std::memory_order_acquire);
        if (t >= h) return t - h;
        return buffer_capacity - h + t;
    }

    size_t capacity() const { 
        return buffer_capacity - 1; 
    }

    void clear() {
        head.store(0, std::memory_order_release);
        tail.store(0, std::memory_order_release);
    }
};

}
#endif   