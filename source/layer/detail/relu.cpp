//
// Created by Mason on 2024/11/1.
//

#include <layer/detail/relu.hpp>
#include "layer/abstract/layer_factory.hpp"
#include "omp.h"

namespace BatmanInfer {
    InferStatus ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
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
                LOG(ERROR) << "The input tensor array in the relu layer has an empty tensor "
                           << i << " th";
                return InferStatus::bInferFailedInputEmpty;
            }
            if (output_data != nullptr && !output_data->empty()) {
                if (input_data->shapes() != output_data->shapes()) {
                    LOG(ERROR) << "The input and output tensor shapes of the relu "
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
//            CHECK(input == nullptr || !input->empty())
//                            << "The input tensor array in the relu layer has an empty tensor " << i
//                            << " th";

            auto output = outputs.at(i);
//                DLOG(ERROR)
//                        << "The output tensor array in the relu layer has an empty tensor "
//                        << i << " th";

#pragma omp critical
                {
                    if (output == nullptr || output->empty()) {
                        output = std::make_shared<Tensor<float>>(input->shapes());
                        outputs.at(i) = output;
                    }
                }

            CHECK(output->shapes() == input->shapes())
                            << "The input and output tensor shapes of the relu layer do not match "
                            << i << " th";
#pragma omp simd
            for (uint32_t j = 0; j < input->size(); ++j) {
                output->index(j) = std::max(input->index(j), 0.0f);
            }
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus ReluLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                    std::shared_ptr<Layer> &relu_layer) {
        CHECK(op != nullptr) << "Relu operator is nullptr";
        relu_layer = std::make_shared<ReluLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    // 使用工具类注册算子
    LayerRegistererWrapper bReluGetInstance("Relu", ReluLayer::GetInstance);
}
