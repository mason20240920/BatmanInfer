//
// Created by Mason on 2025/4/23.
//
#include <thread>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>


#include "cpu/kernels/assembly/bi_arm_gemm.hpp"
#include "runtime/bi_scheduler.hpp"
#include "utils/utils.hpp"

#include <arm_neon.h>
#include <stdio.h>

#include "data/core/neon/bi_ne_asymm.hpp"

namespace GemmOutputStageTest {
    void print_int32x4_t(int32x4_t v) {
        int32_t tmp[4];
        vst1q_s32(tmp, v); // 将 NEON 寄存器中的数据存到普通数组
        printf("[%d, %d, %d, %d]\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    }
}

TEST(GemmOutputStagePerChannel, TestChannelQATVal) {
    // 输入数据，16个int32
    int32_t input_data[16] = {
        1000, -2000, 3000, -4000,
        5000, -6000, 7000, -8000,
        9000, -10000, 11000, -12000,
        13000, -14000, 15000, -16000
    };

    // 将输入数据分成4组，每组4个，装到int32x4x4_t
    int32x4x4_t in_s32;
    in_s32.val[0] = vld1q_s32(&input_data[0]);
    in_s32.val[1] = vld1q_s32(&input_data[4]);
    in_s32.val[2] = vld1q_s32(&input_data[8]);
    in_s32.val[3] = vld1q_s32(&input_data[12]);


    int result_fixedpoint_multiplier = 1819209472;
    int32_t result_shift = 7;
    //  返回值类型：int32x4_t（4 个 32 位整数组成的向量）每个向量为10 [10, 10, 10, 10]
    int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(10);
    int8x16_t min_s8 = vdupq_n_s8(-128); // v 现在等于 [-128, ....] x16个元素
    int8x16_t max_s8 = vdupq_n_s8(127); // [127, ....]
    bool is_bounded_relu = false;

    int8x16_t out_s8 = BatmanInfer::finalize_quantization(
        in_s32,
        result_fixedpoint_multiplier,
        result_shift,
        result_offset_after_shift_s32,
        min_s8,
        max_s8,
        is_bounded_relu
    ); // 存储输出
    int8_t output_data[16];
    vst1q_s8(output_data, out_s8);

    // 打印结果
    for (int i = 0; i < 16; ++i) {
        printf("%d ", static_cast<int>(output_data[i]));
    }
    printf("\n");
}
