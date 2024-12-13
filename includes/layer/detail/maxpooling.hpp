//
// Created by Mason on 2024/11/5.
//

#ifndef BATMAN_INFER_MAXPOOLING_HPP
#define BATMAN_INFER_MAXPOOLING_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class MaxPoolingLayer: public NonParamLayer {
    public:
        /**
         * 最大池化
         * @param padding_h 填充高度
         * @param padding_w 填充宽度
         * @param pooling_size_h 池化窗口高度
         * @param pooling_size_w 池化窗口宽度
         * @param stride_h 步长高度
         * @param stride_w 步长宽度
         */
        explicit MaxPoolingLayer(uint32_t padding_h,
                                 uint32_t padding_w,
                                 uint32_t pooling_size_h,
                                 uint32_t pooling_size_w,
                                 uint32_t stride_h,
                                 uint32_t stride_w);


        InferStatus Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                            std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
                const std::shared_ptr<RuntimeOperator>& op,
                std::shared_ptr<Layer>& max_layer);

    private:
        uint32_t padding_h_ = 0;
        uint32_t padding_w_ = 0;
        uint32_t pooling_size_h_ = 0;
        uint32_t pooling_size_w_ = 0;
        uint32_t stride_h_ = 1;
        uint32_t stride_w_ = 1;
    };
}

#endif //BATMAN_INFER_MAXPOOLING_HPP
