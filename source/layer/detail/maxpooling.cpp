//
// Created by Mason on 2024/11/5.
//

#include <layer/detail/maxpooling.hpp>
#include "layer/abstract/layer_factory.hpp"
#include <Halide.h>

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

    InferStatus MaxPoolingLayer::Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                                         std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output array size of the max pooling layer "
                          "do not match";
            return InferStatus::bInferFailedOutputSizeError;
        }

        const uint32_t pooling_h = pooling_size_h_;
        const uint32_t pooling_w = pooling_size_w_;
        if (!stride_h_ || !stride_w_) {
            LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                          "greater than 0";
            return InferStatus::bInferFailedStrideParameterError;
        }

        // First loop to check inputs and set up outputs
        auto output_iter = outputs.begin();
        for (const auto&[input_name, input_tensor]: inputs) {
            if (input_tensor == nullptr || input_tensor->empty()) {
                LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                           << "empty tensor "
                           << input_name;
                continue;
            } else {
                uint32_t input_h = input_tensor->rows();
                uint32_t input_w = input_tensor->cols();
                auto output_h = uint32_t(std::floor((int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
                auto output_w = uint32_t(std::floor(int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1);
                if (!output_h || !output_w) {
                    LOG(ERROR) << "The output size of tensor" << input_name << "th"
                               << " in the max pooling layer is less than zero";
                    continue;
                } else {
                    std::shared_ptr<ftensor>& output_data = output_iter->second;
                    if (output_data != nullptr && !output_data->empty()) {
                        if (output_data->rows() != output_h ||
                            output_data->cols() != output_w) {
                            LOG(ERROR) << "The output tensor array in the max pooling layer "
                                       << "has an incorrectly sized tensor "
                                       << input_name;
                            continue;
                        }
                    } else {
                        // Allocate output tensor if not already allocated
                        output_data = std::make_shared<Tensor<float>>(
                                input_tensor->channels(),
                                output_h,
                                output_w);
                    }
                }
            }
            output_iter++;
        }

        output_iter = outputs.begin();
        for (const auto&[input_name, input_tensor]: inputs) {
            if (input_tensor->empty()) {
                LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                              "empty tensor " << input_name << "th";
                continue;
            }

            // 定义输入和输出的 Halide 张量
            Halide::Buffer<float> input(input_tensor->data());

            // 确定维度数
            int dimensions = input.dimensions();

            // 输入矩阵的高度
            const int input_h = static_cast<int>(input_tensor->rows());
            // 输入矩阵的宽度
            const int input_w = static_cast<int>(input_tensor->cols());

            // 输出的矩阵数据
            std::shared_ptr<Tensor<float>>& output_data = output_iter->second;

            CHECK(output_data != nullptr && !output_data->empty())
                            << "The output tensor array in the max pooling layer "
                               "has an incorrectly sized tensor "
                            << input_name << "th";

            // 每个channel进行池化
            Halide::Buffer<float> output(output_data->data());

            // 定义 Halide 变量
            std::vector<Halide::Var> vars(dimensions);
            for (int i = 0; i < dimensions; ++i)
                vars[i] = Halide::Var("dim" + std::to_string(i));


            const int pooling_w_halide = static_cast<int>(pooling_w);
            const int pooling_h_halide = static_cast<int>(pooling_h);

            // 输入填充函数
            Halide::Func padded_input("padded_input");

            // 构建参数列表
            std::vector<Halide::Expr> args;
            args.reserve(dimensions);
            for (int i = 0; i < dimensions; ++i)
                args.push_back(vars[i]);

            // 构建条件
            std::vector<Halide::Expr> conditions;
            for (int i = 0; i < dimensions; ++i)
                if (i == 0 || i == 1)  // 空间维度
                    conditions.push_back(vars[i] >= 0 && vars[i] < input_tensor->data().dim[i].extent);

            // 组合所有条件
            Halide::Expr all_conditions = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i)
                all_conditions = all_conditions && conditions[i];


            // 构建输入参数
            std::vector<Halide::Expr> input_args;
            for (int i = 0; i < dimensions; ++i) {
                if (i == 0 || i == 1) {
                    input_args.push_back(clamp(vars[i], 0, input_tensor->data().dim[i].extent-1));
                } else {
                    input_args.push_back(vars[i]);
                }
            }

            padded_input(args) = select(all_conditions,
                                        input(input_args),
                                        0.0f);
            // 定义 Halide 函数
            Halide::RDom r(0, pooling_w_halide, 0, pooling_h_halide);
            Halide::Func pool("pool");

            // 构建池化函数的参数
            std::vector<Halide::Expr> pool_args = args;  // 复制原始参数列表
            // 只修改最后两个维度（空间维度）
            pool_args[0] = vars[0] * static_cast<int>(stride_w_) + r.x - static_cast<int>(padding_w_);  // width维度
            pool_args[1] = vars[1] * static_cast<int>(stride_h_) + r.y - static_cast<int>(padding_h_);  // height维度

            // 修改池化操作
            pool(args) = maximum(padded_input(pool_args));

            // 计算输出
            pool.realize(output);

            output.copy_to_host();
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

    LayerRegistererWrapper bMaxPoolingGetInstance("MaxPool",
                                                  MaxPoolingLayer::GetInstance);
}