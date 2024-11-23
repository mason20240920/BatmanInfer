//
// Created by Mason on 2024/11/21.
//
#include <layer/detail/concat.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    ConcatLayer::ConcatLayer(int axis) : NonParamLayer("Concat"), axis_(axis) {

    }

    InferStatus ConcatLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the Concat layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != 2) {
            LOG(ERROR) << "The input tensor array size of the Concat "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        // channel, row, cols合并
        if (axis_ != 0 && outputs.size() != 1) {
            LOG(ERROR) << "The Add layer requires exactly one output tensor";
            return InferStatus::bInferFailedOutputSizeError;
        }

        // Batch size合并-
        if (axis_ == 0 && outputs.size() != inputs.size()) {
            LOG(ERROR) << "The Concat layer requires exactly same output tensor";
            return InferStatus::bInferFailedOutputSizeError;
        }

        // 获取input的size
        const uint32_t num_inputs = inputs.size();
        // 获取一个batch里面的[channel, row, cols]
        const auto &first_input = inputs.at(0);

        if (first_input == nullptr || first_input->empty()) {
            LOG(ERROR) << "The first input tensor in the Concat layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        // [channel, row, cols]
        const auto& first_shape = first_input->shapes();
        if (axis_ < 0 || axis_ >= static_cast<int>(first_shape.size()) + 1) {
            LOG(ERROR) << "Invalid axis for Concat Layer: " << axis_;
            return InferStatus::bInferFailedInputShapeError;
        }

        // 如果是batch size合并，只需要判断input的tensor的shape是否一致
        // 检查所有输入张量的形状是否一致（除了拼接轴）
        bool shape_mismatch = false;

        // 通过 shared(shape_mismatch) 共享变量，用于标记是否存在形状不匹配
#pragma omp parallel for shared(shape_mismatch)
        for (uint32_t i = 1; i < num_inputs; ++i) {
            // 如果已经发现不匹配，跳过检查
            if (shape_mismatch) continue;

            const auto &input = inputs.at(i);
            // 验证input里面的Tensor是否都存在
            if (input == nullptr || input->empty()) {
#pragma omp critical
                LOG(ERROR) << "One of the input tensors in the Concat layer is empty";
                shape_mismatch = true;
                return InferStatus::bInferFailedInputEmpty;
            }

            // 验证inputs里面每个张量形状是否一致
            const auto shape = input->shapes();
            // 如果axis_是0，走batch size合并，否则走[C, W, H]合并
            if (axis_ == 0) {
                if (shape != first_shape) {
#pragma omp critical
                    LOG(ERROR) << "Input tensor batch size shapes do not match for Concat layer";
                    shape_mismatch = true;
                }
                break;
            } else {
                for (size_t dim = 0; dim < shape.size(); ++dim) {
                    if (dim != static_cast<size_t>(axis_ - 1) && shape[dim] != first_shape[dim]) {
#pragma omp critical
                        LOG(ERROR) << "Input tensor shapes do not match for Concat layer";
                        shape_mismatch = true;
                    }
                    break;
                }
            }
        }
        if (shape_mismatch)
            return InferStatus::bInferFailedInputShapeError;

        // 如果axis_设置为0: 一般是Batch size合并，
        // 而不是channel合并
        if (axis_ == 0) {
            merge_tensors(inputs, outputs);
            return InferStatus::bInferFailedOutputEmpty;
        }

        // 使用 Concat静态方法拼接输入张量: axis_要减去1: 表示从N, C, W, H中C开始进行合并
        auto output_tensor = Concat(inputs, axis_ - 1);

        if (!output_tensor) {
            LOG(ERROR) << "Failed to concatenate tensors in Concat Layer";
            return InferStatus::bInferFailedOutputEmpty;
        }

        // 拼接结果赋值给输出
        outputs[0] = output_tensor;

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus ConcatLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                                std::shared_ptr<Layer> &concat_layer) {
        CHECK(op != nullptr) << "Concat operator is nullptr";
        const auto &params = op->params;

        if (params.find("axis") == params.end()) {
            LOG(ERROR) << "Can not find the axis parameter";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        auto axis = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("axis"));
        if (axis == nullptr) {
            LOG(ERROR) << "Axis parameter is invalid";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        concat_layer = std::make_shared<ConcatLayer>(axis->value);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bConcatCreateInstance("Concat", ConcatLayer::CreateInstance);

}