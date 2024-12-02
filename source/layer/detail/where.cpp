//
// Created by Mason on 2024/12/2.
//

#include <layer/detail/where.hpp>
#include <layer/abstract/layer_factory.hpp>

namespace BatmanInfer {
    InferStatus WhereLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
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