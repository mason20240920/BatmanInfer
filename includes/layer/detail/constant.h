//
// Created by Mason on 2024/11/27.
//

#ifndef BATMANINFER_CONSTANT_H
#define BATMANINFER_CONSTANT_H

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class ConstantLayer: public NonParamLayer {
    public:
        /**
         * 定义常量的值，后期需要优化
         * @param val
         */
        explicit ConstantLayer(std::vector<float> value);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                       std::shared_ptr<Layer>& constant_layer);

    private:
        std::vector<float> value_;
    };
}

#endif //BATMANINFER_CONSTANT_H
