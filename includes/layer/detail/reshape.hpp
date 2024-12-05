//
// Created by Mason on 2024/12/5.
//

#ifndef BATMANINFER_RESHAPE_HPP
#define BATMANINFER_RESHAPE_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class ReshapeLayer: public NonParamLayer {
    public:
        explicit ReshapeLayer(int allowzero);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
                            std::vector<std::shared_ptr<Tensor<float> > > &outputs) override;


        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &reshape_layer);

    private:
        int allowzero_;
    };
}

#endif //BATMANINFER_RESHAPE_HPP
