//
// Created by Mason on 2024/11/27.
//

#include <layer/detail/equal.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    InferStatus EqualLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the flatten layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() - 1 != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the flatten "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        // 获取常量参数
        float constant_val = inputs.at(1)->at(0, 0, 0);

        // 获取常量的值
//        auto constant_val = inputs.at(0)

        const uint32_t batch_size = inputs.size() - 1;
        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
            auto &output = outputs.at(i);

            if (input == nullptr || input->empty()) {
                LOG(ERROR) << "The input tensor in the Flatten layer is empty";
                return InferStatus::bInferFailedInputEmpty;
            }

            // [channel, row, col]
            const auto input_shape = input->shapes();
            if (input_shape.size() != 3) {
                LOG(ERROR) << "Each input tensor must have 3 dimensions [channel, row, col]";
                return InferStatus::bInferFailedInputShapeError;
            }

            output = TensorClone(input);
            CHECK(input->size() == output->size())
                            << "The output and input shapes of the flatten layer do "
                               "not match "
                            << i << " th";
            output->Equal(constant_val);

            outputs.at(i) = output;
        }
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus EqualLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                        std::shared_ptr<Layer> &equal_layer) {
        CHECK(op != nullptr) << "Equal operator is nullptr";

        equal_layer = std::make_shared<EqualLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bEqualCreateInstance("Equal", EqualLayer::CreateInstance);
}