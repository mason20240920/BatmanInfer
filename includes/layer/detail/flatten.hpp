//
// Created by Mason on 2024/11/14.
//

#ifndef BATMAN_INFER_FLATTEN_HPP
#define BATMAN_INFER_FLATTEN_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class FlattenLayer: public NonParamLayer {
    public:
        /**
         * 平铺的坐标轴
         * @param axis
         */
        explicit FlattenLayer(int axis);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                       std::shared_ptr<Layer>& flatten_layer);

    private:
        int axis_ = 1;
    };
}

#endif //BATMAN_INFER_FLATTEN_HPP
