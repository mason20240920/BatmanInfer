//
// Created by Mason on 2024/10/18.
//

#ifndef BATMAN_INFER_TENSOR_UTIL_HPP
#define BATMAN_INFER_TENSOR_UTIL_HPP

#include <data/Tensor.hpp>
#include <onnx/onnx.pb.h>
#include <runtime/runtime_datatype.hpp>
#include "runtime/runtime_parameter.hpp"

namespace BatmanInfer {

    /**
     * 创建一个张量
     * @param channels 通道数量
     * @param rows 行数
     * @param cols 列数
     * @return
     */
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t batch_size,
                                                uint32_t channels,
                                                uint32_t rows,
                                                uint32_t cols);

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

    /**
     * 返回一个深拷贝后的张量
     * @param tensor 待拷贝的张量
     * @return 新的张量
     */
    std::shared_ptr<Tensor<float>> TensorClone(const std::shared_ptr<Tensor<float>> &tensor);

    /**
     * 矩乘: 两个Tensor进行矩乘
     * @param tensor1
     * @param tensor2
     * @return
     */
    std::shared_ptr<Tensor<float>> MatrixMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                                                  const std::shared_ptr<Tensor<float>> &tensor2);

    /**
     * @brief 按元素相乘
     * @param tensor1 张量1
     * @param tensor2 张量2
     * @return 返回按元素相乘的结果
     */
    std::shared_ptr<Tensor<float>> MultiplyElement(const std::shared_ptr<Tensor<float>> &tensor1,
                                                   const std::shared_ptr<Tensor<float>> &tensor2);

    /**
     * 拼接函数
     * @param tensors
     * @param axis 沿着哪个轴进行合并
     * @return
     */
    std::shared_ptr<Tensor<float>> Concat(const std::vector<std::shared_ptr<Tensor<float>>> &tensors,
                                          int axis);

    /***
     * 把两个batch的inputs输入到一个outputs里面
     * @param tensors
     * @param merge_tensor
     */
    void merge_tensors(const std::vector<sftensor> &tensors,
                       std::vector<sftensor> &merge_tensor);

    /**
     * 对张量进行下三角或上三角置为0
     * @param tensor
     * @param upper
     * @return
     */
    std::shared_ptr<Tensor<float>> Trilu(const std::shared_ptr<Tensor<float>> &tensor,
                                         int upper);


    /**
     * @brief 数据结构转换
     * @param runtime_parameter_int
     * @return
     */
    std::vector<float> convert_to_int_vector(const RuntimeParameterIntArray *runtime_parameter_int);
}

#endif //BATMAN_INFER_TENSOR_UTIL_HPP
