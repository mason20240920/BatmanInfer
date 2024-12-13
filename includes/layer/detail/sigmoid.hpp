//
// Created by Mason on 2024/11/5.
//

#ifndef BATMAN_INFER_SIGMOID_HPP
#define BATMAN_INFER_SIGMOID_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class SigmoidLayer : public NonParamLayer {
    public:
        SigmoidLayer() : NonParamLayer("Softmax") {}

        InferStatus Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                            std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
                const std::shared_ptr<RuntimeOperator> &op,
                std::shared_ptr<Layer> &sigmoid_layer);
    };
}

#endif //BATMAN_INFER_SIGMOID_HPP
