//
// Created by Mason on 2024/12/4.
//

#include <layer/abstract/layer_factory.hpp>
#include <layer/detail/transpose.hpp>
#include <data/tensor_util.hpp>

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
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the transpose layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output array size of the transpose layer "
                          "do not match";
            return InferStatus::bInferFailedOutputSizeError;
        }

        const uint32_t batch = inputs.size();
        for (uint32_t i = 0; i < batch; i++) {
            const auto& input = inputs.at(i);
            auto& output = outputs.at(i);

            auto temp = TensorClone(input);
            temp->Reshape(output->shapes(), true);
            output = temp;
        }
        return InferStatus::bInferSuccess;
    }

    LayerRegistererWrapper bTransposeGetInstance("Transpose", TransposeLayer::CreateInstance);
}