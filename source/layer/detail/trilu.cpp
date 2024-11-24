//
// Created by Mason on 2024/11/24.
//

#include <layer/detail/trilu.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    TriluLayer::TriluLayer(int upper) : NonParamLayer("Trilu"), upper_(upper) {}

    InferStatus TriluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the trilu layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the trilu "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        const uint32_t batch_size = inputs.size();

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

            output = Trilu(input, upper_);
            CHECK(input->size() == output->size())
                            << "The output and input shapes of the trilu layer do "
                               "not match "
                            << i << " th";
            outputs.at(i) = output;
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus TriluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                        std::shared_ptr<Layer> &trilu_layer) {
        CHECK(op != nullptr) << "Trilu operator is nullptr";
        const auto& params = op->params;

        if (params.find("upper") == params.end()) {
            LOG(ERROR) << "Can not find the upper parameter";
            return ParseParameterAttrStatus::bParameterMissingUpper;
        }

        auto upper = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("upper"));

        if (upper == nullptr)
            return ParseParameterAttrStatus::bParameterMissingUpper;

        trilu_layer = std::make_shared<TriluLayer>(upper->value);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bTriluCreateInstance("Trilu", TriluLayer::CreateInstance);
}