//
// Created by Mason on 2024/11/13.
//

#include <layer/detail/add.hpp>
#include <layer/abstract/layer_factory.hpp>
#include "omp.h"

namespace BatmanInfer {
    InferStatus AddLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.size() != 2) {
            LOG(ERROR) << "The Add layer requires exactly two input tensors";
            return InferStatus::bInferFailedInputSizeError;
        }

        if (inputs.empty() || outputs.empty()) {
            LOG(ERROR) << "The input or output tensor array in the Add layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (outputs.size() != 1) {
            LOG(ERROR) << "The Add layer requires exactly one output tensor";
            return InferStatus::bInferFailedOutputSizeError;
        }

        const std::shared_ptr<Tensor<float>> &input1 = inputs.at(0);
        const std::shared_ptr<Tensor<float>> &input2 = inputs.at(1);
        const std::shared_ptr<Tensor<float>> &output = outputs.at(0);

        if (input1 == nullptr || input1->empty() || input2 == nullptr || input2->empty()) {
            LOG(ERROR) << "One of the input tensors in the add layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (input1->shapes() != input2->shapes()) {
            LOG(ERROR) << "The input tensor shapes of the Add layer do not match";
            return InferStatus::bInferFailedOutputSizeError;
        }

        // Parallelize the element-wise addition
#pragma omp parallel for
        for (uint32_t j = 0; j < input1->size(); ++j)
            output->index(j) = input1->index(j) + input2->index(j);

        return InferStatus::bInferSuccess;

    }

    ParseParameterAttrStatus AddLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                   std::shared_ptr<Layer> &add_layer) {
        CHECK(op != nullptr) << "Add operator is nullptr";
        add_layer = std::make_shared<AddLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bAddGetInstance("Add", AddLayer::GetInstance);
}