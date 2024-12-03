//
// Created by Mason on 2024/12/3.
//

#include <data/tensor_util.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <layer/detail/sqrt.hpp>

namespace BatmanInfer {
    InferStatus SqrtLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
        std::vector<std::shared_ptr<Tensor<float> > > &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the sqrt layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the sqrt "
                          "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        const uint32_t batch_size = inputs.size();

        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
            auto &output = outputs.at(i);

            if (input == nullptr || input->empty()) {
                LOG(ERROR) << "The input tensor in the Flatten layer is empty";
                return InferStatus::bInferFailedInputEmpty;
            }

            output = TensorClone(input);
            output->Sqrt();
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus SqrtLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &sqrt_layer) {
        CHECK(op != nullptr) << "Sqrt operator is nullptr";

        sqrt_layer = std::make_shared<SqrtLayer>();

        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bSqrtCreateInstance("Sqrt", SqrtLayer::GetInstance);


}