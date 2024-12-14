//
// Created by Mason on 2024/11/7.
//

#ifndef BATMAN_INFER_CONVOLUTION_HPP
#define BATMAN_INFER_CONVOLUTION_HPP
#include <layer/abstract/param_layer.hpp>

namespace BatmanInfer {
    class ConvolutionLayer : public ParamLayer {
    public:
        explicit ConvolutionLayer(uint32_t output_channel,
                                  uint32_t in_channel,
                                  uint32_t kernel_h,
                                  uint32_t kernel_w,
                                  uint32_t padding_t,
                                  uint32_t padding_l,
                                  uint32_t padding_b,
                                  uint32_t padding_r,
                                  uint32_t stride_h,
                                  uint32_t stride_w,
                                  uint32_t groups,
                                  bool use_bias = true);

        static ParseParameterAttrStatus GetInstance(
                const std::shared_ptr<RuntimeOperator>& op,
                std::shared_ptr<Layer>& conv_layer);

        InferStatus Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                            std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) override;

        /**
         * 初始化kernel的im2col排布
         */
        void InitIm2ColWeight();

    private:
        void ConvGemmBias(const arma::fmat& input_matrix,
                          sftensor output_tensor,
                          uint32_t group,
                          uint32_t kernel_index,
                          uint32_t kernel_count_group,
                          const arma::frowvec& kernel,
                          uint32_t output_w,
                          uint32_t output_h) const;

        /**
         * 卷积计算转为: 矩阵计算 (现有较为成熟的矩阵)
         * 将一个nxn的矩阵转为 1 x (nxn)的行向量
         * @param input: 输入特征图像
         * @param kernel_w: 卷积核宽度
         * @param kernel_h: 卷积核高度
         * @param input_w: 输入特征的宽度
         * @param input_h: 输入特征的高度
         * @param input_c_group: 每个group处理的通道数量
         * @param group: 当前Im2Col的组数(Group)
         * @param row_len: 卷积核展开后的列数
         * @param col_len: 卷积计算的次数
         * @return
         */
        halide_buffer_t Im2Row(const sftensor& input,
                               uint32_t kernel_w,
                               uint32_t kernel_h,
                               uint32_t input_w,
                               uint32_t input_h,
                               uint32_t input_c_group,
                               uint32_t group,
                               uint32_t row_len,
                               uint32_t col_len) const;

    private:
        bool use_bias_ = false;
        uint32_t groups_ = 1;
        uint32_t top_padding_ = 0;
        uint32_t left_padding_ = 0;
        uint32_t bottom_padding_ = 0;
        uint32_t right_padding_ = 0;
        uint32_t stride_h_ = 1;
        uint32_t stride_w_ = 1;
        std::vector<arma::frowvec> kernel_matrix_arr_;
    };
}

#endif //BATMAN_INFER_CONVOLUTION_HPP
