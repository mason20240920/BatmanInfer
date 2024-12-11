//
// Created by Mason on 2024/11/1.
//

#ifndef BATMAN_INFER_RELU_HPP
#define BATMAN_INFER_RELU_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class ReluLayer : public NonParamLayer {
        public:
        ReluLayer(): NonParamLayer("Relu") {}

        InferStatus Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                            std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
                const std::shared_ptr<RuntimeOperator>& op,
                std::shared_ptr<Layer>& relu_layer);
    };
}

#endif //BATMAN_INFER_RELU_HPP
