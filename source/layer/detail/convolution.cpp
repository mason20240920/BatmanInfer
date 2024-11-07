//
// Created by Mason on 2024/11/7.
//

#include <layer/detail/convolution.hpp>
#include <glog/logging.h>
#include <layer/abstract/layer_factory.hpp>

namespace BatmanInfer{
    arma::fmat ConvolutionLayer::Im2Col(BatmanInfer::sftensor input,
                                        uint32_t kernel_w,
                                        uint32_t kernel_h,
                                        uint32_t input_w,
                                        uint32_t input_h,
                                        uint32_t input_c_group,
                                        uint32_t group,
                                        uint32_t row_len,
                                        uint32_t col_len) const {
        // 存储对输入图像展开后的矩阵
        arma::fmat input_matrix(input_c_group * row_len, col_len);
        const uint32_t input_padded_h = input_h + 2 * padding_h_;
        const uint32_t input_padded_w = input_w + 2 * padding_w_;
        const float padding_value = 0.f;
        // 提取当前的输入通道, 将该通道起始值指针赋值给input_channel_ptr
        for (uint32_t ic = 0; ic < input_c_group; ++ic) {
            // ic: 通道的索引:
            float* input_channel_ptr = input->matrix_raw_ptr(ic + group * input_c_group);
            // 当前的列
            uint32_t current_col = 0;
            // 当前的行：展开通道后开始摆放的起始位置
            uint32_t channel_row = ic * row_len;
            for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += stride_w_) {
                for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += stride_h_) {
                    float* input_matrix_ptr = input_matrix.colptr(current_col) + channel_row;
                    current_col += 1;
                }
            }
        }
    }
}