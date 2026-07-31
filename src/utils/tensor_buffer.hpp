// pre allocs memory once
// resuable buffer
// zero copy
// reduces memory frag

#ifndef YOLO_TENSOR_BUFFER_HPP
#define YOLO_TENSOR_BUFFER_HPP

#include <vector>
#include <cstdint>
#include "types.hpp"

namespace yolo {

//pre alloc tensor buffer. avoids repeated alloc
class TensorBuffer {
private:
    std::vector<float> data;
    std::vector<int64_t> shape;
    size_t total_elements;
    int input_size;

public:
    explicit TensorBuffer(const ModelConfig& config)
        : input_size(config.input_size) {
        shape = {1, 3, input_size, input_size};
        total_elements = 1 * 3 * input_size * input_size;
        data.resize(total_elements);
    }

    // get raw data pointer
    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }

    // get shape
    const int64_t* getShape() const { return shape.data(); }
    size_t getShapeSize() const { return shape.size(); }

    // get total elements
    size_t size() const { return total_elements; }

    // get input size
    int getInputSize() const { return input_size; }

    // reset all vals to 0
    void reset() {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    //copy data from external source
    void copyFrom(const float* source, size_t count) {
        size_t copy_size = std::min(count, total_elements);
        // num. of bytes to copy over
        std::memcpy(data.data(), source, copy_size * sizeof(float));
    }

};

}

#endif