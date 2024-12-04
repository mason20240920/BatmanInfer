//
// Created by Mason on 2024/12/4.
//

#include <layer/abstract/layer_factory.hpp>
#include <layer/detail/mul.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    InferStatus MulLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the mul layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }
        if (inputs.size() != 2) {
            LOG(ERROR) << "The input  tensor array size of the mul layer do "
                       << "not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        if (outputs.size() != 1) {
            LOG(ERROR) << "The output tensor of Mul is error";
            return InferStatus::bInferFailedOutputSizeError;
        }

        auto& output = outputs.at(0);
        output = MultiplyElement(inputs.at(0), inputs.at(1));

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    MulLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &mul_layer) {
        CHECK(op != nullptr) << "Add operator is nullptr";
        mul_layer = std::make_shared<MulLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bMulCreateInstance("Mul", MulLayer::CreateInstance);
}