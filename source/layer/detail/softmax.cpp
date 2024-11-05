//
// Created by Mason on 2024/11/4.
//

#include <layer/detail/softmax.hpp>
#include <layer/abstract/layer_factory.hpp>
#include "omp.h"

namespace BatmanInfer {
    InferStatus SoftmaxLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the softmax layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }
        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the softmax layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
            const std::shared_ptr<Tensor<float>> &output_data = outputs.at(i);
            if (input_data == nullptr || input_data->empty()) {
                LOG(ERROR) << "The input tensor at index " << i << " is empty in the softmax layer";
                return InferStatus::bInferFailedInputEmpty;
            }
            if (output_data != nullptr && !output_data->empty()) {
                if (input_data->shapes() != output_data->shapes()) {
                    LOG(ERROR) << "The input and output tensor shapes at index " << i
                               << " do not match in the softmax layer";
                    return InferStatus::bInferFailedInputOutSizeMatchError;
                }
            }
        }

        // 并行处理批次中的每个样本
#pragma omp parallel for
        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
            auto output = outputs.at(i);

            // 确保输出张量已正确初始化
#pragma omp critical
            {
                if (output == nullptr || output->empty()) {
                    output = std::make_shared<Tensor<float>>(input->shapes());
                    outputs.at(i) = output;
                }
            }

            CHECK(output->shapes() == input->shapes())
                            << "The input and output tensor shapes at index " << i << " do not match in the softmax layer";

            // 获取输入张量的形状
            const std::vector<uint32_t> &input_shape = input->raw_shapes();
            uint32_t dim_size = input_shape.size();
            // Just make dim is 2. make it follows the col direction
            auto dim = static_cast<int>(dim_size - 1);

            // 处理负数的维度值
            int actual_dim = dim;
            if (dim < 0) {
                actual_dim = dim + dim_size;
            }

            // 检查维度是否有效
            if (actual_dim < 0 || actual_dim >= static_cast<int>(dim_size)) {
                LOG(ERROR) << "Invalid dimension " << dim << " for softmax computation";
                continue;
            }

            // 计算沿指定维度的大小
            uint32_t axis_size = input_shape[actual_dim];

            // 计算除指定维度外的总元素数 N
            uint32_t N = 1;
            for (uint32_t d = 0; d < input_shape.size(); ++d) {
                if (static_cast<int>(d) != actual_dim) {
                    N *= input_shape[d];
                }
            }

            // 分配临时缓冲区用于存储最大值和指数和
            std::vector<float> max_vals(N, -std::numeric_limits<float>::infinity());
            std::vector<float> sum_exps(N, 0.0f);

            // 计算每个切片沿指定维度的最大值
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t k = 0; k < axis_size; ++k) {
                    // 计算在原始张量中的索引
                    uint32_t flat_index = n * axis_size + k;
                    std::vector<uint32_t> idx = input->unravel_index(flat_index, input_shape);
                    float val = input->at(idx);
                    if (val > max_vals[n]) {
                        max_vals[n] = val;
                    }
                }
            }

            // 计算指数并累加和
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t k = 0; k < axis_size; ++k) {
                    uint32_t flat_index = n * axis_size + k;
                    std::vector<uint32_t> idx = input->unravel_index(flat_index, input_shape);
                    float exp_val = std::exp(input->at(idx) - max_vals[n]);
                    output->at(idx) = exp_val;
                    sum_exps[n] += exp_val;
                }
            }

            // 归一化得到最终的 softmax 概率
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t k = 0; k < axis_size; ++k) {
                    uint32_t flat_index = n * axis_size + k;
                    std::vector<uint32_t> idx = input->unravel_index(flat_index, input_shape);
                    output->at(idx) /= sum_exps[n];
                }
            }
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    SoftmaxLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                              std::shared_ptr<Layer> &softmax_layer) {
        CHECK(op != nullptr) << "Softmax operator is nullptr";
        softmax_layer = std::make_shared<SoftmaxLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    // 使用工具类注册算子
    LayerRegistererWrapper bSoftmaxGetInstance("nn.Softmax", SoftmaxLayer::GetInstance);


}