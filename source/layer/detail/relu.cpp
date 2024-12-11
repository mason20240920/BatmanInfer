//
// Created by Mason on 2024/11/1.
//

#include <layer/detail/relu.hpp>
#include "layer/abstract/layer_factory.hpp"
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    InferStatus ReluLayer::Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                                   std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the relu layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }
        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                       << "not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        // 根据索引进行查询
        auto input_iter = inputs.begin();
        auto output_iter = outputs.begin();


        for (int i = 0; i < inputs.size(); ++i) {
            output_iter->second = TensorClone(input_iter->second);
            output_iter->second->Relu();
            ++input_iter;
            ++output_iter;
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
