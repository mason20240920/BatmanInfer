//
// Created by Mason on 2024/12/6.
//

#include <layer/detail/split.hpp>
#include <layer/abstract/layer_factory.hpp>

namespace BatmanInfer {
    InferStatus SplitLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    SplitLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                               std::shared_ptr<Layer> &split_layer) {
        CHECK(op != nullptr) << "Split Operator is nullptr";

        split_layer = std::make_shared<SplitLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bSplitCreateInstance("Split", SplitLayer::CreateInstance);
}