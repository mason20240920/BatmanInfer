//
// Created by Mason on 2024/11/27.
//

#ifndef BATMANINFER_EQUAL_HPP
#define BATMANINFER_EQUAL_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class EqualLayer: public NonParamLayer {
    public:
        EqualLayer():NonParamLayer("Equal") {}

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                       std::shared_ptr<Layer>& equal_layer);
    };
}

#endif //BATMANINFER_EQUAL_HPP
