//
// Created by Mason on 2025/1/13.
//

#ifndef BATMANINFER_KERNEL_WEIGHT_FORMAT_HPP
#define BATMANINFER_KERNEL_WEIGHT_FORMAT_HPP

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

namespace BatmanGemm {

    /* 内部枚举，用于定义内核所期望的权重格式。
     *
     * 该枚举与 bi_arm_gemm.hpp 中定义的“外部”WeightFormat 不同，主要是为了支持 SVE（可扩展向量扩展）。
     * 在 SVE 中，内核的定义基于 SVE 向量长度的倍数，而外部格式会根据运行设备的向量长度（VL）转换为固定格式。
     * 编码为一个位字段（bitfield）：
     *
     * 第 0 位：SVE 标志。
     * 第 4 位：BF16 转换标志（快速模式）。
     * 第 8-11 位：块长度（以字节为单位）。
     * 第 12-15 位：向量数量。
     */
    enum class KernelWeightFormat {
        NON_FIXED = 0,
        VL128_BL16 = 0x1200,
        VL128_BL32 = 0x1400,
        VL128_BL32_BF16 = 0x1410,
        VL128_BL64 = 0x1800,
        VL256_BL64 = 0x2800,
        VL256_BL64_BF16 = 0x2810,
        VL1VL_BL16 = 0x1201,
        VL1VL_BL32 = 0x1401,
        VL1VL_BL32_BF16 = 0x1411,
        VL1VL_BL64 = 0x1801,
        VL2VL_BL64 = 0x2801,
        VL2VL_BL64_BF16 = 0x2811
    };

    WeightFormat get_weight_format(const KernelWeightFormat, size_t);

} // namespace arm_gemm

#endif //BATMANINFER_KERNEL_WEIGHT_FORMAT_HPP
