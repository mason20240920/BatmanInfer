//
// Created by Mason on 2024/12/2.
//

#include <data/tensor_util.hpp>
#include <layer/detail/cast.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <others/utils.hpp>

namespace BatmanInfer {
    CastLayer::CastLayer(RuntimeDataType dtype): NonParamLayer("Cast"), dtype_(dtype) {

    }


    InferStatus CastLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
        std::vector<std::shared_ptr<Tensor<float> > > &outputs) {

        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the cast layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the cast "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        // TODO: implements the cast detail, float32 to float16
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
        }


        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus CastLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &cast_layer) {
        CHECK(op != nullptr) << "Equal operator is nullptr";

        const auto& params = op->params;

        if (params.find("to") == params.end()) {
            LOG(ERROR) << "Can not find the to parameter";
            return ParseParameterAttrStatus::bParameterMissingUnknown;
        }

        auto to = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("to"));

        if (to == nullptr) {
            LOG(ERROR) << "Can not find the to parameter";
            return ParseParameterAttrStatus::bParameterMissingUnknown;
        }

        cast_layer = std::make_shared<CastLayer>(convert_runtime_data(to->value));

        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bCastLayerCreateInstance("Cast", CastLayer::CreateInstance);

}
