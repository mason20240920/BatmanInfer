//
// Created by Mason on 2024/11/13.
//

#include <layer/detail/global_average_pool.hpp>
#include <layer/abstract/layer_factory.hpp>
#include "omp.h"

namespace BatmanInfer {
    // Confirm this one is just one batch size, not multi batches size
    InferStatus GlobalAveragePoolLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                                std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.size() != 1) {
            LOG(ERROR) << "The GlobalAveragePool layer requires exactly one input tensor";
            return InferStatus::bInferFailedInputSizeError;
        }
        if (outputs.size() != 1) {
            LOG(ERROR) << "The GlobalAveragePool layer requires exactly one output tensor";
            return InferStatus::bInferFailedOutputSizeError;
        }

        const std::shared_ptr<Tensor<float>> &input = inputs.at(0);
        auto &output = outputs.at(0);

        if (input == nullptr || input->empty()) {
            LOG(ERROR) << "The input tensor in the GlobalAveragePool layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        const auto input_shape = input->shapes();
        if (input_shape.size() != 3) {
            LOG(ERROR) << "The input tensor must be 3-dimensional [C, H, W]";
            return InferStatus::bInferFailedInputSizeError;
        }

        const uint32_t batch_size = inputs.size();
        const uint32_t channels = input_shape[0];
        const uint32_t height = input_shape[1];
        const uint32_t width = input_shape[2];
        const uint32_t pool_size = height * width;

        // Output shape should be [N, C, 1, 1]
        // N - batch size
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<float>>(std::vector<uint32_t>{channels, 1, 1});
            outputs.at(0) = output;
        }

        if (output->shapes() != std::vector<uint32_t>{channels, 1, 1}) {
            LOG(ERROR) << "The output tensor shape of the GlobalAveragePool layer does not match expected shape [C, 1, 1]";
            return InferStatus::bInferFailedOutputSizeError;
        }

        // Perform global average pooling
#pragma omp parallel for collapse(2)
        for (uint32_t n = 0; n < batch_size; ++n) {
            for (uint32_t c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (uint32_t h = 0; h < height; ++h) {
                    for (uint32_t w = 0; w < width; ++w) {
                        sum += input->at(c, h, w);
                    }
                }
                output->at(c, 0, 0) = sum / pool_size;
            }
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus GlobalAveragePoolLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                                 std::shared_ptr<Layer> &global_avg_pool_layer) {
        CHECK(op != nullptr) << "GlobalAveragePool operator is nullptr";
        global_avg_pool_layer = std::make_shared<GlobalAveragePoolLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bGlobalAveragePoolGetInstance("GlobalAveragePool", GlobalAveragePoolLayer::GetInstance);
}