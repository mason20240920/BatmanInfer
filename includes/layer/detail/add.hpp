//
// Created by Mason on 2024/11/13.
//

#ifndef BATMAN_INFER_ADD_HPP
#define BATMAN_INFER_ADD_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class AddLayer : public NonParamLayer {
    public:
        AddLayer() : NonParamLayer("Add") {}
        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                    std::shared_ptr<Layer>& add_layer);
    };
}

#endif //BATMAN_INFER_ADD_HPP
