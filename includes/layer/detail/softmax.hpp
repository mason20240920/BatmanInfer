//
// Created by Mason on 2024/11/4.
//

#ifndef BATMAN_INFER_SOFTMAX_HPP
#define BATMAN_INFER_SOFTMAX_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class SoftmaxLayer: public NonParamLayer {
    public:
        SoftmaxLayer(): NonParamLayer("Softmax") {}
        InferStatus Forward(
                const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus GetInstance(
                const std::shared_ptr<RuntimeOperator>& op,
                std::shared_ptr<Layer>& softmax_layer);
    };
}

#endif //BATMAN_INFER_SOFTMAX_HPP