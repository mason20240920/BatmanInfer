//
// Created by Mason on 2024/11/5.
//

#include <layer/detail/maxpooling.hpp>
#include "layer/abstract/layer_factory.hpp"
#include "omp.h"

namespace BatmanInfer {
    MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h,
                                     uint32_t padding_w,
                                     uint32_t pooling_size_h,
                                     uint32_t pooling_size_w,
                                     uint32_t stride_h,
                                     uint32_t stride_w) : NonParamLayer("MaxPooling"),
                                     padding_h_(padding_h),
                                     padding_w_(padding_w),
                                     pooling_size_h_(pooling_size_h),
                                     pooling_size_w_(pooling_size_w),
                                     stride_h_(stride_h),
                                     stride_w_(stride_w) {}

    InferStatus MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output array size of the max pooling layer "
                          "do not match";
            return InferStatus::bInferFailedOutputSizeError;
        }

        const uint32_t batch = inputs.size();
        const uint32_t pooling_h = pooling_size_h_;
        const uint32_t pooling_w = pooling_size_w_;
        if (!stride_h_ || !stride_w_) {
            LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                          "greater than 0";
            return InferStatus::bInferFailedStrideParameterError;
        }

        // First loop to check inputs and set up outputs
        // 输入和输出尺寸的计算, 计算公式在README.md
#pragma omp parallel for
        for (uint32_t i = 0; i < batch; ++i) {
            const std::shared_ptr<ftensor>& input_data = inputs.at(i);
            if (input_data == nullptr || input_data->empty()) {
                LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                           << "empty tensor "
                           << i << "th";
                continue;
            } else {
                uint32_t input_h = input_data->rows();
                uint32_t input_w = input_data->cols();
                auto output_h = uint32_t(std::floor((int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
                auto output_w = uint32_t(std::floor(int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1);
                if (!output_h || !output_w) {
                    LOG(ERROR) << "The output size of tensor" << i << "th"
                               << " in the max pooling layer is less than zero";
                    continue;
                } else {
                    std::shared_ptr<ftensor>& output_data = outputs.at(i);
                    if (output_data != nullptr && !output_data->empty()) {
                        if (output_data->rows() != output_h ||
                            output_data->cols() != output_w) {
                            LOG(ERROR) << "The output tensor array in the max pooling layer "
                                       << "has an incorrectly sized tensor "
                                       << i << "th";
                            continue;
                        }
                    } else {
                        // Allocate output tensor if not already allocated
                        output_data = std::make_shared<Tensor<float>>(
                                input_data->channels(),
                                output_h,
                                output_w);
                    }
                }
            }
        }

        // Main computation loop
#pragma omp parallel for
        for (uint32_t i = 0; i < batch; ++i) {
            // 验证一个batch里面的输入是否为空
            const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
            if (input_data == nullptr || input_data->empty()) {
                LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                              "empty tensor " << i << "th";
                continue;
            }

            // 输入矩阵的高度
            const uint32_t input_h = input_data->rows();
            // 输入矩阵的宽度
            const uint32_t input_w = input_data->cols();
            // 输入矩阵(padding)之后的高度
            const uint32_t input_padded_h = input_h + 2 * padding_h_;
            // 输入矩阵(padding)之后的宽度
            const uint32_t input_padded_w = input_w + 2 * padding_w_;
            // 输入矩阵的(channel)的数量(进行池化的次数)
            const uint32_t input_c = input_data->channels();

            // 输出矩阵的高度
            const auto output_h = uint32_t(
                    std::floor((int(input_padded_h) - int(pooling_h)) / stride_h_ + 1));
            // 输出矩阵的宽度
            const auto output_w = uint32_t(
                    std::floor((int(input_padded_w) - int(pooling_w)) / stride_w_ + 1));

            // 输出的矩阵数据
            std::shared_ptr<Tensor<float>>& output_data = outputs.at(i);

            CHECK(output_data != nullptr && !output_data->empty())
                            << "The output tensor array in the max pooling layer "
                               "has an incorrectly sized tensor "
                            << i << "th";

            // 每个channel进行池化
            for (uint32_t ic = 0; ic < input_c; ++ic) {
                const arma::fmat& input_channel = input_data->slice(ic);
                arma::fmat& output_channel = output_data->slice(ic);

                uint32_t total_output_elements = output_h * output_w;

#pragma omp parallel for
                for (uint32_t idx = 0; idx < total_output_elements; ++idx) {
                    // 计算输出矩阵中当前元素的行和列位置
                    uint32_t output_row = idx / output_w;
                    uint32_t output_col = idx % output_w;

                    // 计算输入矩阵中池化窗口的起始列和行位置
                    // This is where the pooling window will be applied
                    uint32_t c = output_col * stride_w_;
                    uint32_t r = output_row * stride_h_;

                    // 获取指向输出矩阵当前列的指针，用于后续的最大值赋值
                    float* output_channel_ptr = output_channel.colptr(output_col);

                    // 初始化 max_value 为最小可能值，以便在池化窗口中寻找最大值
                    float max_value = std::numeric_limits<float>::lowest();

                    // 遍历池化窗口的每一列，并检查列索引是否超出输入矩阵的范围。
                    for (uint32_t w = 0; w < pooling_w; ++w) {
                        // Calculate the column index in the input tensor
                        const uint32_t col_idx = c + w - padding_w_;

                        // Skip if the column index is out of bounds
                        if (col_idx >= input_w) continue;

                        // Get a pointer to the current column in the input channel
                        const float* col_ptr = input_channel.colptr(col_idx);

                        // Iterate over each row in the pooling window
                        for (uint32_t h = 0; h < pooling_h; ++h) {
                            // Calculate the row index in the input tensor
                            const uint32_t row_idx = r + h - padding_h_;

                            // Skip if the row index is out of bounds
                            if (row_idx >= input_h) continue;

                            // Get the value at the current position in the input tensor
                            float current_value = *(col_ptr + row_idx);

                            // Update max_value if the current value is greater
                            if (current_value > max_value) {
                                max_value = current_value;
                            }
                        }
                    }
                    // Set the maximum value found in the pooling window to the current position in the output tensor
                    *(output_channel_ptr + output_row) = max_value;
                }
            }
        }
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus MaxPoolingLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                          std::shared_ptr<Layer> &max_layer) {
        CHECK(op != nullptr) << "Maxpooling get instance failed, operator is nullptr";
        const std::map<std::string, std::shared_ptr<RuntimeParameter>> &params = op->params;
        // 是否包含步数
        if (params.find("strides") == params.end()) {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::bParameterMissingStride;
        }

        auto stride = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("strides"));
        if (!stride) {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::bParameterMissingStride;
        }

        if (params.find("pads") == params.end()) {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::bParameterMissingPadding;
        }

        auto padding = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("pads"));
        if (!padding) {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::bParameterMissingPadding;
        }

        if (params.find("kernel_shape") == params.end()) {
            LOG(ERROR) << "Can not find the kernel shape parameter";
            return ParseParameterAttrStatus::bParameterMissingKernel;
        }

        auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("kernel_shape"));
        if (!kernel_size) {
            LOG(ERROR) << "Can not find the kernel size parameter";
            return ParseParameterAttrStatus::bParameterMissingKernel;
        }
        const auto& padding_values = padding->value;
        const auto& stride_values = stride->value;
        const auto& kernel_values = kernel_size->value;

        const uint32_t dims = 2;
        if (padding_values.size() != 4) {
            LOG(ERROR) << "Can not find the right padding parameter";
            return ParseParameterAttrStatus::bParameterMissingPadding;
        }

        if (stride_values.size() != dims) {
            LOG(ERROR) << "Can not find the right stride parameter";
            return ParseParameterAttrStatus::bParameterMissingStride;
        }

        if (kernel_values.size() != dims) {
            LOG(ERROR) << "Can not find the right kernel size parameter";
            return ParseParameterAttrStatus::bParameterMissingKernel;
        }

        max_layer = std::make_shared<MaxPoolingLayer>(
                padding_values.at(0),
                padding_values.at(1),
                kernel_values.at(0),
                kernel_values.at(1),
                stride_values.at(0),
                stride_values.at(1));

        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

//    LayerRegistererWrapper bMaxPoolingGetInstance("MaxPool",
//                                                  MaxPoolingLayer::GetInstance);
}