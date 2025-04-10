//
// Created by holynova on 25-4-8.
//

#pragma once

#include "neon/neon_defines.h"

namespace BatmanInfer {

    /** 计算两个浮点型的、每个含 4x2 个元素的向量 a、b 对应通道的最大值
     *
     * @param[in] a 浮点型输入向量
     * @param[in] b 浮点型输入向量
     *
     * @return 新的向量，存储每个对应通道的最大值
     */
    float32x4x2_t vmax2q_f32(float32x4x2_t a, float32x4x2_t b);

} // namespace BatmanInfer
#include "data/core/neon/bi_neon_fixed_point.inl"
