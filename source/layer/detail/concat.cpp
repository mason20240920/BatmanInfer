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

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the Concat "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        // 获取input的size
        const uint32_t num_inputs = inputs.size();
        const auto &first_input = inputs.at(0);

        if (first_input == nullptr || first_input->empty()) {
            LOG(ERROR) << "The first input tensor in the Concat layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        const auto first_shape = first_input->shapes();
        if (axis_ < 0 || axis_ >= static_cast<int>(first_shape.size())) {
            LOG(ERROR) << "Invalid axis for Concat Layer: " << axis_;
            return InferStatus::bInferFailedInputShapeError;
        }


        // 检查所有输入张量的形状是否一致（除了拼接轴）
        for (uint32_t i = 1; i < num_inputs; ++i) {
            const auto &input = inputs.at(i);
            if (input == nullptr || input->empty()) {
                LOG(ERROR) << "One of the input tensors in the Concat layer is empty";
                return InferStatus::bInferFailedInputEmpty;
            }

            const auto shape = input->shapes();
            for (size_t dim = 0; dim < shape.size(); ++dim) {
                if (dim != static_cast<size_t>(axis_) && shape[dim] != first_shape[dim]) {
                    LOG(ERROR) << "Input tensor shapes do not match for Concat layer";
                    return InferStatus::bInferFailedInputShapeError;
                }
            }
        }

        // 使用 Concat静态方法拼接输入张量
        auto output_tensor = Concat(inputs, axis_);

        if (!output_tensor) {
            LOG(ERROR) << "Failed to concatenate tensors in Concat Layer";
            return InferStatus::bInferFailedOutputEmpty;
        }

        // 拼接结果赋值给输出
        outputs[0] = output_tensor;

        return InferStatus::bInferSuccess;
    }
}