//
// Created by Mason on 2024/12/4.
//

#include <layer/detail/matmul.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    InferStatus MatMulLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor of MatMul is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != 2) {
            LOG(ERROR) << "The input tensor of MatMul is error";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (outputs.size() != 1) {
            LOG(ERROR) << "The output tensor of MatMul is error";
            return InferStatus::bInferFailedOutputSizeError;
        }

        auto& output = outputs.at(0);
        output = MatrixMultiply(inputs.at(0), inputs.at(1));

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    MatMulLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &matmul_layer) {
        if (op == nullptr) {
            LOG(ERROR) << "MatMul operator is nullptr";
            return ParseParameterAttrStatus::bParameterMissingValue;
        }

        matmul_layer = std::make_shared<MatMulLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bMatMulCreateInstance("MatMul", MatMulLayer::CreateInstance);
}