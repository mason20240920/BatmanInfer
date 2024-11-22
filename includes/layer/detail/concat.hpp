//
// Created by Mason on 2024/11/21.
//

#ifndef BATMANINFER_CONCAT_HPP
#define BATMANINFER_CONCAT_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class ConcatLayer: public NonParamLayer {
    public:
        /**
         * 合并的轴
         * @param axis: 0: channel, 1: row, 2: cols
         */
        explicit ConcatLayer(int axis);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                       std::shared_ptr<Layer>& concat_layer);

    private:
        // 合并的轴
        int axis_ = 1;
    };
}

#endif //BATMANINFER_CONCAT_HPP
