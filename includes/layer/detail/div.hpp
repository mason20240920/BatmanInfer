//
// Created by Mason on 2024/12/4.
//

#ifndef BATMAN_INFER_DIV_HPP
#define BATMAN_INFER_DIV_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class DivLayer final : public NonParamLayer {
      public:
        DivLayer(): NonParamLayer("Div") {}
        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                    std::shared_ptr<Layer>& div_layer);
    };
}

#endif //BATMAN_INFER_DIV_HPP
