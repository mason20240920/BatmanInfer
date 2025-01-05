//
// Created by Mason on 2025/1/5.
//

#ifndef BATMANINFER_BI_CONVOLUTION_PARAMETERS_HPP
#define BATMANINFER_BI_CONVOLUTION_PARAMETERS_HPP

#pragma once

#include <cstdint>

namespace BatmanGemm {
    /*
     * 参数集用于描述“卷积”类型的 GEMM（通用矩阵乘法）。
     *
     * 这个结构的作用是动态完成卷积操作，而无需提前对输入张量进行 im2row 展开。
     * 参数中包含卷积的核心信息（如输入输出张量形状、卷积核大小、填充方式等），
     * 并支持动态生成展开矩阵，以节省内存和计算资源。
     *
     * 参数说明：
     * - 卷积参数：描述输入输出张量的形状、卷积核大小、填充方式以及量化张量的填充值（如果需要）。
     * - 内存布局：假设输入张量是 NHWC 格式，提供基地址和步幅信息（列、行、批次的步长）来描述张量在内存中的存储方式。
     *
     * 限制：
     * - 不支持“multis”参数（可能是多分量张量或特定优化策略）。
     *
     * 通过这些参数，可以直接在计算过程中动态完成卷积操作，无需额外的内存展开步骤。
     */
    struct BIConvolutionParameters {
        int64_t input_width;
        int64_t input_height;
        int64_t input_channels;
        int64_t kernel_width;
        int64_t kernel_height;
        int64_t output_width;
        int64_t output_height;
        int64_t output_stride_w;
        int64_t output_stride_h;
        //          output_channels not included as they do not affect the input.
        int64_t dilation_w;
        int64_t dilation_h;
        int64_t padding_top;
        int64_t padding_left;
        float   padding_value;
    };
}

#endif //BATMANINFER_BI_CONVOLUTION_PARAMETERS_HPP
