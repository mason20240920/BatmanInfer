//
// Created by Mason on 2024/12/4.
//

#ifndef BATMANINFER_MUL_HPP
#define BATMANINFER_MUL_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class MulLayer: public NonParamLayer {
    public:
        MulLayer(): NonParamLayer("Mul") {}

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
                            std::vector<std::shared_ptr<Tensor<float> > > &outputs) override;


        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &mul_layer);
    };
}

#endif //BATMANINFER_MUL_HPP
