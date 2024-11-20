//
// Created by Mason on 2024/11/5.
//

#include <layer/detail/sigmoid.hpp>
#include <layer/abstract/layer_factory.hpp>
#include "omp.h"

namespace BatmanInfer {
    InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the relu layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }
        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                       << "not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            const sftensor &input_data = inputs.at(i);
            const sftensor &output_data = outputs.at(i);
            if (input_data == nullptr || input_data->empty()) {
                LOG(ERROR) << "The input tensor array in the sigmoid layer has an empty tensor "
                           << i << " th";
                return InferStatus::bInferFailedInputEmpty;
            }
            if (output_data != nullptr && !output_data->empty()) {
                if (input_data->shapes() != output_data->shapes()) {
                    LOG(ERROR) << "The input and output tensor shapes of the sigmoid "
                                  "layer do not match "
                               << i << " th";
                    return InferStatus::bInferFailedInputOutSizeMatchError;
                }
            }
        }

        // Parallelize the outer loop for batch processing
#pragma omp parallel for
        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
            auto output = outputs.at(i);

#pragma omp critical
            {
                if (output == nullptr || output->empty()) {
                    output = std::make_shared<Tensor<float>>(input->shapes());
                    outputs.at(i) = output;
                }
            }

            CHECK(output->shapes() == input->shapes())
            << "The input and output tensor shapes of the sigmoid layer do not match at index "
            << i;

            // Use OpenMP SIMD for vectorization
#pragma omp simd
            for (uint32_t j = 0; j < input->size(); ++j) {
                // Compute the sigmoid function
                output->index(j) = 1.0f / (1.0f + expf(-input->index(j)));
            }
        }
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus SigmoidLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &sigmoid_layer) {
        CHECK(op != nullptr) << "Sigmoid operator is nullptr";
        sigmoid_layer = std::make_shared<SigmoidLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    // 使用工具类注册算子
    LayerRegistererWrapper bSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);
}