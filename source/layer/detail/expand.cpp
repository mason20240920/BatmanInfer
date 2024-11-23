//
// Created by Mason on 2024/11/23.
//

#include <layer/detail/expand.hpp>
#include "layer/abstract/layer_factory.hpp"
#include "omp.h"

namespace BatmanInfer {
    InferStatus ExpandLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the expand layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the expand layer do "
                       << "not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        // 判断batch是否为空
        bool batch_mismatch = false;

        const uint32_t batch_size = inputs.size();
#pragma omp parallel for shared(batch_mismatch)
        for (uint32_t i = 0; i < batch_size; ++i) {
            if (batch_mismatch) continue;

            const sftensor &input_data = inputs.at(i);
            const sftensor &output_data = outputs.at(i);
            if (input_data == nullptr || input_data->empty()) {
#pragma omp critical
                LOG(ERROR) << "The input tensor array in the expand layer has an empty tensor "
                           << i << " th";
                batch_mismatch = true;
            }
            if (output_data == nullptr || output_data->empty()) {
#pragma omp critical
                LOG(ERROR) << "The output tensor array in the expand layer has an empty tensor "
                           << i << " th";
                batch_mismatch = true;
            }
        }

        if (batch_mismatch)
            return InferStatus::bInferFailedInputEmpty;

        // 进行扩张
#pragma omp parallel for
        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
            auto& output = outputs.at(i);

            input->Expand(output->shapes());
            output = input;
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus ExpandLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &expand_layer) {
        CHECK(op != nullptr) << "Expand operator is nullptr";
        expand_layer = std::make_shared<ExpandLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bExpandGetInstance("Expand", ExpandLayer::GetInstance);
}