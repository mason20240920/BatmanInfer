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

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the flatten "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        const uint32_t batch_size = inputs.size();
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