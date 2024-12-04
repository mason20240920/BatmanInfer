//
// Created by Mason on 2024/12/4.
//

#ifndef BATMANINFER_MATMUL_HPP
#define BATMANINFER_MATMUL_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class MatMulLayer: public NonParamLayer {
    public:
        MatMulLayer(): NonParamLayer("MatMul") {}

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
                            std::vector<std::shared_ptr<Tensor<float> > > &outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &matmul_layer);
    };
}

#endif //BATMANINFER_MATMUL_HPP
