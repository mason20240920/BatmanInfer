//
// Created by Mason on 2024/12/2.
//

#ifndef BATMANINFER_WHERE_HPP
#define BATMANINFER_WHERE_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class WhereLayer: public NonParamLayer {
    public:
        WhereLayer(): NonParamLayer("Where") {}

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                       std::shared_ptr<Layer>& where_layer);
    };
}

#endif //BATMANINFER_WHERE_HPP
