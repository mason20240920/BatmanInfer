//
// Created by Mason on 2024/12/6.
//

#ifndef BATMANINFER_SPLIT_HPP
#define BATMANINFER_SPLIT_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class SplitLayer : public NonParamLayer {
    public:
        SplitLayer(): NonParamLayer("Split") {}

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
                            std::vector<std::shared_ptr<Tensor<float> > > &outputs) override;


        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &split_layer);
    };
}

#endif //BATMANINFER_SPLIT_HPP
