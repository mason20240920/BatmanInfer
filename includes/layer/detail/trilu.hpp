//
// Created by Mason on 2024/11/24.
//

#ifndef BATMANINFER_TRILU_HPP
#define BATMANINFER_TRILU_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class TriluLayer: public NonParamLayer {
    public:
        /**
         *
         * @param upper: 0: 下三角操作
         *               1: 上三角操作
         */
        explicit TriluLayer(int upper);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                       std::shared_ptr<Layer>& trilu_layer);

    private:
        int upper_;
    };
}

#endif //BATMANINFER_TRILU_HPP
