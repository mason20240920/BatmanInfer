//
// Created by Mason on 2024/12/2.
//

#include <layer/detail/where.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    InferStatus WhereLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the where layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != 3) {
            LOG(ERROR) << "The input tensor array in the where layer is wrong size";
            return InferStatus::bInferFailedInputShapeError;
        }

        const auto& x_val = inputs.at(1)->at({0, 0, 0});
        const auto& y_val = inputs.at(2)->at({0, 0, 0});

        const uint32_t batch_size = inputs.size() - 2;

        for (uint32_t i = 0; i < batch_size; i++) {
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

            output->Where(x_val, y_val);

        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus WhereLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                        std::shared_ptr<Layer> &where_layer) {
        CHECK(op != nullptr) << "Where operator is nullptr";

        where_layer = std::make_shared<WhereLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bWhereCreateInstance("Where", WhereLayer::CreateInstance);
}