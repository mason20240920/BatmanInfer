//
// Created by Mason on 2024/11/14.
//

#ifndef BATMAN_INFER_GEMM_HPP
#define BATMAN_INFER_GEMM_HPP

#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace BatmanInfer {
    class GemmLayer : public ParamLayer {
    public:
        explicit GemmLayer(float alpha,
                           float beta,
                           int32_t trans_b,
                           int32_t row,
                           int32_t cols);
        static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                             std::shared_ptr<Layer> &gemm_layer);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
    private:
        float alpha_ = 1;
        float beta_ = 1;
        int32_t trans_b_ = 1;
        int gemm_height_;
        int gemm_width_;
    };

}

#endif //BATMAN_INFER_GEMM_HPP
