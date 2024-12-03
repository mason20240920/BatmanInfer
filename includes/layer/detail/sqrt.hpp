//
// Created by Mason on 2024/12/3.
//

#ifndef BATMAN_INFER_SQRT_HPP
#define BATMAN_INFER_SQRT_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class SqrtLayer : public NonParamLayer {
    public:
      SqrtLayer(): NonParamLayer("Sqrt") {}
      InferStatus Forward(
                  const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                  std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus GetInstance(
                const std::shared_ptr<RuntimeOperator>& op,
                std::shared_ptr<Layer>& sqrt_layer);
    };
}

#endif //BATMAN_INFER_SQRT_HPP
