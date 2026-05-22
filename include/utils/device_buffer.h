#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include "error_check.h"

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : size_(count * sizeof(T)) {
        if (size_ > 0) {
            HIP_CHECK(hipMalloc(&ptr_, size_));
        } else {
            ptr_ = nullptr;
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            (void)hipFree(ptr_); 
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                (void)hipFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }

    operator T*() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};