//
// Created by Mason on 2024/10/18.
//

#include <glog/logging.h>
#include <data/Tensor.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels,
                                                uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(channels, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(1, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size) {
        return std::make_shared<Tensor<float>>(1, 1, size);
    }

    std::shared_ptr<Tensor<float>> TensorClone(std::shared_ptr<Tensor<float>> tensor) {
        return std::make_shared<Tensor<float>>(*tensor);
    }
}