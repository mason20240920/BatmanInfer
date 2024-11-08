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
        // 计算填充后的输入特征图尺寸
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
                    // 遍历卷积核的宽度方向
                    // 例子: 对于一个 2 x 2 的卷积核，这个循环会执行两次，分别为kw = 0, kw = 1
                    for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                        // 计算当前卷积核元素在输入特征图中的水平偏移量
                        // 例子: 例如w = 0, padding_w = 0, 当kw = 0时候, region_w 为 3 x (0 + 0 - 0) = 0
                        // 当kw = 1时, 3 x (0 + 1 - 0) = 3
                        const uint32_t region_w = input_h * (w + kw - padding_w_);
                        // 遍历卷积核的高度方向
                        // 例子: 对于一个 2 x 2 的卷积核，这个循环会执行两次，分别为kh = 0, kh = 1
                         for(uint32_t kh = 0; kh < kernel_h; ++kh) {
                             // 检查当前卷积核位置是否在有效的输入特征图范围内
                             if ((kh + r >= padding_h_ && kw + w >= padding_w_) &&
                                 (kh + r < input_h + padding_h_ &&
                                  kw + w < input_w + padding_w_)) {
                                 float* region_ptr =
                                         input_channel_ptr + region_w + (r + kh - padding_h_);
                                 *input_matrix_ptr = *region_ptr;
                             } else {
                                 // 如果卷积核位置超出输入特征图的边界，使用填充值（例如0）
                                 *input_matrix_ptr = padding_value;  // only support zero mode
                             }
                             input_matrix_ptr += 1;
                         }
                    }
                }
            }
        }
        return  input_matrix;
    }

    ConvolutionLayer::ConvolutionLayer(uint32_t output_channel,
                                       uint32_t in_channel,
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t padding_h,
                                       uint32_t padding_w,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t groups,
                                       bool use_bias): ParamLayer("Convolution"),
                                       use_bias_(use_bias),
                                       groups_(groups),
                                       padding_h_(padding_h),
                                       padding_w_(padding_w),
                                       stride_h_(stride_h),
                                       stride_w_(stride_w) {
        if (groups != 1)
            in_channel /= groups;
        this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
        if (use_bias)
            this->InitBiasParam(output_channel, 1, 1, 1);
    }

    InferStatus ConvolutionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                          std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the convolution layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the convolution "
                       << "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        if (weights_.empty()) {
            LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                          "be greater than zero";
            return InferStatus::bInferFailedWeightParameterError;
        }

        if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
            LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
            return InferStatus::bInferFailedBiasParameterError;
        }

        if (!stride_h_ || !stride_w_) {
            LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                          "greater than 0";
            return InferStatus::bInferFailedStrideParameterError;
        }

        // 卷积核的总数量(关注特征不同)
        const uint32_t kernel_count = this->weights_.size();
        // 单个卷积核的高度
        const uint32_t kernel_h = this->weights_.at(0)->rows();
        // 单个卷积核的宽度
        const uint32_t kernel_w = this->weights_.at(0)->cols();
        // 单个卷积核的通道数（与输入特征必须一致）
        const uint32_t kernel_c = this->weights_.at(0)->channels();
        // 单个卷积核的一个通道展开成一行后的长度
        const uint32_t row_len = kernel_h * kernel_w;
        CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
              << "The size of kernel matrix in the convolution layer should be greater "
              << "than zero";

        // 关注不同的卷积核的height, width, channel是否一致
        for (uint32_t k = 0; k < kernel_count; ++k) {
            const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
            CHECK(kernel->rows() == kernel_h);
            CHECK(kernel->cols() == kernel_w);
            CHECK(kernel->channels() == kernel_c);
        }
        // 表示每个组内的卷积核数量 (每个组内的卷积核数量)
        // 分组卷积将输入通道和卷积核分为多个组，每个组只在内部进行卷积操作。这样可以减少计算量和参数数量
        const uint32_t kernel_count_group = kernel_count / groups_;
        const uint32_t batch_size = inputs.size();

        if (kernel_matrix_arr_.empty()) {
            if (groups_ == 1)
                CHECK(kernel_matrix_arr_.size() == kernel_count_group)
                << "The number of kernel matrix and kernel_count_group do not match";
            else
                CHECK(kernel_matrix_arr_.size() == kernel_count)
                << "The number of kernel matrix and kernel_count do not match";
        }
    }
}