//
// Created by Mason on 2024/11/14.
//

#include <layer/detail/flatten.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    FlattenLayer::FlattenLayer(int axis) :
    NonParamLayer("Flatten"), axis_(axis) {}

    InferStatus FlattenLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the flatten layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the flatten "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        int axis = axis_;

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

            output = TensorClone(input);
            CHECK(input->size() == output->size())
                            << "The output and input shapes of the flatten layer do "
                               "not match "
                            << i << " th";
            outputs.at(i) = output;

            const uint32_t channels = input_shape[0];
            const uint32_t rows = input_shape[1];
            const uint32_t cols = input_shape[2];
            const uint32_t flattened_size = channels * rows * cols;

            if (output == nullptr || output->empty()) {
                output = std::make_shared<Tensor<float>>(std::vector<uint32_t>{flattened_size});
                outputs.at(i) = output;
            } else {
                // Flatten to 1D
                output->Reshape({flattened_size}, true);
            }

            if (output->shapes() != std::vector<uint32_t>{1, 1, flattened_size}) {
                LOG(ERROR) << "The output tensor shape of the Flatten layer does not match expected shape";
                return InferStatus::bInferFailedOutputShapeError;
            }
        }
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus FlattenLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                          std::shared_ptr<Layer> &flatten_layer) {
        CHECK(op != nullptr) << "Flatten operator is nullptr";
        const auto& params = op->params;

        if (params.find("axis") == params.end()) {
            LOG(ERROR) << "Can not find the axis parameter";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        auto axis = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("axis"));

        if (axis == nullptr)
            return ParseParameterAttrStatus::bParameterMissingAxis;

        flatten_layer = std::make_shared<FlattenLayer>(axis->value);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bFlattenCreateInstance("Flatten", FlattenLayer::CreateInstance);
}