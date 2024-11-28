//
// Created by Mason on 2024/11/27.
//


#include <layer/detail/constant.h>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {

    InferStatus ConstantLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                       std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus ConstantLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                          std::shared_ptr<Layer> &constant_layer) {
        CHECK(op != nullptr) << "Flatten operator is nullptr";

        constant_layer = std::make_shared<ConstantLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bConstantCreateInstance("Constant", ConstantLayer::CreateInstance);

}