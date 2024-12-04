//
// Created by Mason on 2024/12/4.
//

#include <layer/detail/div.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>
#include "omp.h"

namespace BatmanInfer {
    InferStatus DivLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
        std::vector<std::shared_ptr<Tensor<float> > > &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The Div layer requires exactly input tensors";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != 2) {
            LOG(ERROR) << "The Div layer requires exactly two input tensors";
            return InferStatus::bInferFailedInputSizeError;
        }

        // 除数获取
        const sftensor &div_input = inputs.at(1);
        const float div_simple = div_input->at({0, 0, 0});

        constexpr auto batch_size = 1;

        for (int b = 0; b < batch_size; ++b) {
            const sftensor &div_output = inputs.at(b);
            sftensor &output_data = outputs.at(b);

            if (div_output == nullptr || div_output->empty()) {
                LOG(ERROR) << "The input tensor array in the div layer has an empty tensor "
                           << b << " th";
                return InferStatus::bInferFailedInputEmpty;
            }
            if (output_data != nullptr && !output_data->empty()) {
                if (div_output->shapes() != output_data->shapes()) {
                    LOG(ERROR) << "The input and output tensor shapes of the div "
                                  "layer do not match "
                               << b << " th";
                    return InferStatus::bInferFailedInputOutSizeMatchError;
                }
            }

            output_data = TensorClone(div_output);
            output_data->Div(div_simple);
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus DivLayer::GetInstance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &div_layer) {
        CHECK(op != nullptr) << "Div operator is nullptr";
        div_layer = std::make_shared<DivLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bDivGetInstance("Div", DivLayer::GetInstance);
}