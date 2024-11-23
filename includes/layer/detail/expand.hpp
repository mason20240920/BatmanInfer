//
// Created by Mason on 2024/11/23.
//

#ifndef BATMANINFER_EXPAND_HPP
#define BATMANINFER_EXPAND_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class ExpandLayer: public NonParamLayer {
    public:
        ExpandLayer(): NonParamLayer("Expand") {}
        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                    std::shared_ptr<Layer>& expand_layer);
    };
}

#endif //BATMANINFER_EXPAND_HPP
