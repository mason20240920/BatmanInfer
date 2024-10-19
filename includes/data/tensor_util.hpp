//
// Created by Mason on 2024/10/18.
//

#ifndef BATMAN_INFER_TENSOR_UTIL_HPP
#define BATMAN_INFER_TENSOR_UTIL_HPP

#include <data/Tensor.hpp>

namespace BatmanInfer {

    /**
     * 创建一个张量
     * @param channels 通道数量
     * @param rows 行数
     * @param cols 列数
     * @return
     */
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels,
                                                uint32_t rows,
                                                uint32_t cols);

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows,
                                                uint32_t cols);

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size);
}

#endif //BATMAN_INFER_TENSOR_UTIL_HPP
