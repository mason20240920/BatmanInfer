//
// Created by Mason on 2024/12/4.
//

#include <layer/abstract/layer_factory.hpp>
#include <layer/detail/transpose.hpp>

namespace BatmanInfer {
    TransposeLayer::TransposeLayer(std::vector<int> perms): NonParamLayer("Transpose"), perms_(perms) {

    }

    ParseParameterAttrStatus TransposeLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                            std::shared_ptr<Layer> &transpose_layer) {

        CHECK(op != nullptr) << "Transpose operator is nullptr";

        const std::map<std::string, std::shared_ptr<RuntimeParameter>> &params = op->params;

        if (params.find("perm") == params.end()) {
            LOG(ERROR) << "Can not find the perm parameter";
            return ParseParameterAttrStatus::bParameterMissingPerm;
        }

        auto perm = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("perm"));

        if (!perm) {
            LOG(ERROR) << "Can not find the perm parameter";
            return ParseParameterAttrStatus::bParameterMissingPerm;
        }

        const auto& perm_values = perm->value;

        if (perm_values.size() != 4) {
            LOG(ERROR) << "Can not find the right perm parameter";
            return ParseParameterAttrStatus::bParameterMissingPerm;
        }

        transpose_layer = std::make_shared<TransposeLayer>(perm_values);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    InferStatus TransposeLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        return InferStatus::bInferSuccess;
    }

    LayerRegistererWrapper bTransposeGetInstance("Transpose", TransposeLayer::CreateInstance);
}