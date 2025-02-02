//
// Created by Mason on 2025/1/20.
//
#include <cpu/kernels/bi_cpu_gemm_lowp_matrix_multiply_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_window_helpers.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                void inline vector_matrix_multiply_u8(BIIterator &ina,
                                                      BIIterator &inb,
                                                      BIIterator &out,
                                                      int width_a,
                                                      int width_b,
                                                      int width_out,
                                                      size_t stride_b,
                                                      const BIWindow &window) {
                    execute_window_loop(
                            window,
                            [&](const BICoordinates &id) {
                                if (id.x() > width_b) {
                                    return;
                                }

                                // Note: Since the input are all positives, we can use uint32_t
                                // Accumulators for the block 0
                                uint32x4x4_t c0 = {{vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)}};

                                auto vec_a = reinterpret_cast<const uint8_t *>(ina.ptr());
                                auto matrix_b = reinterpret_cast<const uint8_t *>(inb.ptr());
                                auto vec_a_end_addr = vec_a + width_a;

                                // This for loop performs 8 accumulations
                                for (; vec_a <= (vec_a_end_addr - 8);) {
                                    const uint8x8_t a00_u8 = vld1_u8(vec_a);
                                    const uint8x16_t b00_u8 = vld1q_u8(matrix_b + 0 * stride_b);
                                    const uint8x16_t b10_u8 = vld1q_u8(matrix_b + 1 * stride_b);
                                    const uint8x16_t b20_u8 = vld1q_u8(matrix_b + 2 * stride_b);
                                    const uint8x16_t b30_u8 = vld1q_u8(matrix_b + 3 * stride_b);
                                    const uint8x16_t b40_u8 = vld1q_u8(matrix_b + 4 * stride_b);
                                    const uint8x16_t b50_u8 = vld1q_u8(matrix_b + 5 * stride_b);
                                    const uint8x16_t b60_u8 = vld1q_u8(matrix_b + 6 * stride_b);
                                    const uint8x16_t b70_u8 = vld1q_u8(matrix_b + 7 * stride_b);

                                    // Convert a00_u8 to uint16_t and get the lower part
                                    const uint16x4x2_t a00_u16 = {
                                            {vget_low_u16(vmovl_u8(a00_u8)), vget_high_u16(vmovl_u8(a00_u8))}};

                                    const uint16x4x4_t b00_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))}};

                                    const uint16x4x4_t b10_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b10_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b10_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b10_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b10_u8)))}};

                                    const uint16x4x4_t b20_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b20_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b20_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b20_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b20_u8)))}};

                                    const uint16x4x4_t b30_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b30_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b30_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b30_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b30_u8)))}};

                                    const uint16x4x4_t b40_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b40_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b40_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b40_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b40_u8)))}};

                                    const uint16x4x4_t b50_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b50_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b50_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b50_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b50_u8)))}};

                                    const uint16x4x4_t b60_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b60_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b60_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b60_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b60_u8)))}};

                                    const uint16x4x4_t b70_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b70_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b70_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b70_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b70_u8)))}};

                                    // Accumulate 0:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16.val[0], 0);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16.val[0], 0);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16.val[0], 0);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16.val[0], 0);

                                    // Accumulate 1:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b10_u16.val[0], a00_u16.val[0], 1);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b10_u16.val[1], a00_u16.val[0], 1);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b10_u16.val[2], a00_u16.val[0], 1);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b10_u16.val[3], a00_u16.val[0], 1);

                                    // Accumulate 2:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b20_u16.val[0], a00_u16.val[0], 2);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b20_u16.val[1], a00_u16.val[0], 2);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b20_u16.val[2], a00_u16.val[0], 2);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b20_u16.val[3], a00_u16.val[0], 2);

                                    // Accumulate 3:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b30_u16.val[0], a00_u16.val[0], 3);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b30_u16.val[1], a00_u16.val[0], 3);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b30_u16.val[2], a00_u16.val[0], 3);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b30_u16.val[3], a00_u16.val[0], 3);

                                    // Accumulate 4:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b40_u16.val[0], a00_u16.val[1], 0);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b40_u16.val[1], a00_u16.val[1], 0);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b40_u16.val[2], a00_u16.val[1], 0);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b40_u16.val[3], a00_u16.val[1], 0);

                                    // Accumulate 5:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b50_u16.val[0], a00_u16.val[1], 1);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b50_u16.val[1], a00_u16.val[1], 1);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b50_u16.val[2], a00_u16.val[1], 1);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b50_u16.val[3], a00_u16.val[1], 1);

                                    // Accumulate 6:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b60_u16.val[0], a00_u16.val[1], 2);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b60_u16.val[1], a00_u16.val[1], 2);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b60_u16.val[2], a00_u16.val[1], 2);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b60_u16.val[3], a00_u16.val[1], 2);

                                    // Accumulate 7:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b70_u16.val[0], a00_u16.val[1], 3);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b70_u16.val[1], a00_u16.val[1], 3);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b70_u16.val[2], a00_u16.val[1], 3);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b70_u16.val[3], a00_u16.val[1], 3);

                                    vec_a += 8;
                                    matrix_b += 8 * stride_b;
                                }

                                // This for loop performs the left-over accumulations
                                for (; vec_a < vec_a_end_addr;) {
                                    const uint8x8_t a00_u8 = vld1_dup_u8(vec_a);
                                    const uint8x16_t b00_u8 = vld1q_u8(matrix_b);

                                    const uint16x4x4_t b00_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))}};

                                    // Convert a00_u8 to uint16_t and get the lower part
                                    const uint16x4_t a00_u16 = vget_low_u16(vmovl_u8(a00_u8));

                                    // Accumulate 0:
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16, 0);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16, 0);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16, 0);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16, 0);

                                    vec_a += 1;
                                    matrix_b += stride_b;
                                }

                                auto vec_out = reinterpret_cast<int32_t *>(out.ptr());
                                if (id.x() < (width_out - 16)) {
                                    vst1q_s32(vec_out + 0, vreinterpretq_s32_u32(c0.val[0]));
                                    vst1q_s32(vec_out + 4, vreinterpretq_s32_u32(c0.val[1]));
                                    vst1q_s32(vec_out + 8, vreinterpretq_s32_u32(c0.val[2]));
                                    vst1q_s32(vec_out + 12, vreinterpretq_s32_u32(c0.val[3]));
                                } else {
                                    auto left_over = width_out - id.x();
                                    for (auto k = 0; k < 4 && left_over; ++k) {
                                        for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                            *(vec_out + k * 4 + j) = c0.val[k][j];
                                        }
                                    }
                                }
                            },
                            ina, inb, out);
                }

                void inline vector_matrix_multiply_s8(BIIterator &ina,
                                                      BIIterator &inb,
                                                      BIIterator &out,
                                                      int width_a,
                                                      int width_b,
                                                      int width_out,
                                                      size_t stride_b,
                                                      const BIWindow &window) {
                    execute_window_loop(
                            window,
                            [&](const BICoordinates &id) {
                                if (id.x() > width_b) {
                                    return;
                                }

                                // Accumulators for the block 0
                                int32x4x4_t c0 = {{vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)}};

                                auto vec_a = reinterpret_cast<const int8_t *>(ina.ptr());
                                auto matrix_b = reinterpret_cast<const int8_t *>(inb.ptr());
                                auto vec_a_end_addr = vec_a + width_a;

                                // This for loop performs 8 accumulations
                                for (; vec_a <= (vec_a_end_addr - 8);) {
                                    const int8x8_t a00_s8 = vld1_s8(vec_a);
                                    const int8x16_t b00_s8 = vld1q_s8(matrix_b + 0 * stride_b);
                                    const int8x16_t b10_s8 = vld1q_s8(matrix_b + 1 * stride_b);
                                    const int8x16_t b20_s8 = vld1q_s8(matrix_b + 2 * stride_b);
                                    const int8x16_t b30_s8 = vld1q_s8(matrix_b + 3 * stride_b);
                                    const int8x16_t b40_s8 = vld1q_s8(matrix_b + 4 * stride_b);
                                    const int8x16_t b50_s8 = vld1q_s8(matrix_b + 5 * stride_b);
                                    const int8x16_t b60_s8 = vld1q_s8(matrix_b + 6 * stride_b);
                                    const int8x16_t b70_s8 = vld1q_s8(matrix_b + 7 * stride_b);

                                    // Convert a00_s8 to int16_t and get the lower part
                                    const int16x4x2_t a00_s16 = {
                                            {vget_low_s16(vmovl_s8(a00_s8)), vget_high_s16(vmovl_s8(a00_s8))}};

                                    const int16x4x4_t b00_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))}};

                                    const int16x4x4_t b10_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b10_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b10_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b10_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b10_s8)))}};

                                    const int16x4x4_t b20_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b20_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b20_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b20_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b20_s8)))}};

                                    const int16x4x4_t b30_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b30_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b30_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b30_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b30_s8)))}};

                                    const int16x4x4_t b40_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b40_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b40_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b40_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b40_s8)))}};

                                    const int16x4x4_t b50_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b50_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b50_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b50_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b50_s8)))}};

                                    const int16x4x4_t b60_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b60_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b60_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b60_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b60_s8)))}};

                                    const int16x4x4_t b70_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b70_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b70_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b70_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b70_s8)))}};

                                    // Accumulate 0:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16.val[0], 0);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16.val[0], 0);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16.val[0], 0);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16.val[0], 0);

                                    // Accumulate 1:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b10_s16.val[0], a00_s16.val[0], 1);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b10_s16.val[1], a00_s16.val[0], 1);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b10_s16.val[2], a00_s16.val[0], 1);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b10_s16.val[3], a00_s16.val[0], 1);

                                    // Accumulate 2:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b20_s16.val[0], a00_s16.val[0], 2);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b20_s16.val[1], a00_s16.val[0], 2);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b20_s16.val[2], a00_s16.val[0], 2);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b20_s16.val[3], a00_s16.val[0], 2);

                                    // Accumulate 3:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b30_s16.val[0], a00_s16.val[0], 3);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b30_s16.val[1], a00_s16.val[0], 3);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b30_s16.val[2], a00_s16.val[0], 3);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b30_s16.val[3], a00_s16.val[0], 3);

                                    // Accumulate 4:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b40_s16.val[0], a00_s16.val[1], 0);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b40_s16.val[1], a00_s16.val[1], 0);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b40_s16.val[2], a00_s16.val[1], 0);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b40_s16.val[3], a00_s16.val[1], 0);

                                    // Accumulate 5:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b50_s16.val[0], a00_s16.val[1], 1);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b50_s16.val[1], a00_s16.val[1], 1);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b50_s16.val[2], a00_s16.val[1], 1);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b50_s16.val[3], a00_s16.val[1], 1);

                                    // Accumulate 6:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b60_s16.val[0], a00_s16.val[1], 2);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b60_s16.val[1], a00_s16.val[1], 2);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b60_s16.val[2], a00_s16.val[1], 2);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b60_s16.val[3], a00_s16.val[1], 2);

                                    // Accumulate 7:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b70_s16.val[0], a00_s16.val[1], 3);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b70_s16.val[1], a00_s16.val[1], 3);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b70_s16.val[2], a00_s16.val[1], 3);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b70_s16.val[3], a00_s16.val[1], 3);

                                    vec_a += 8;
                                    matrix_b += 8 * stride_b;
                                }

                                // This for loop performs the left-over accumulations
                                for (; vec_a < vec_a_end_addr;) {
                                    const int8x8_t a00_s8 = vld1_dup_s8(vec_a);
                                    const int8x16_t b00_s8 = vld1q_s8(matrix_b);

                                    const int16x4x4_t b00_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))}};

                                    // Convert a00_s8 to uint16_t and get the lower part
                                    const int16x4_t a00_s16 = vget_low_s16(vmovl_s8(a00_s8));

                                    // Accumulate 0:
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16, 0);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16, 0);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16, 0);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16, 0);

                                    vec_a += 1;
                                    matrix_b += stride_b;
                                }

                                auto vec_out = reinterpret_cast<int32_t *>(out.ptr());
                                if (id.x() < (width_out - 16)) {
                                    vst1q_s32(vec_out + 0, c0.val[0]);
                                    vst1q_s32(vec_out + 4, c0.val[1]);
                                    vst1q_s32(vec_out + 8, c0.val[2]);
                                    vst1q_s32(vec_out + 12, c0.val[3]);
                                } else {
                                    auto left_over = width_out - id.x();
                                    for (auto k = 0; k < 4 && left_over; ++k) {
                                        for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                            *(vec_out + k * 4 + j) = c0.val[k][j];
                                        }
                                    }
                                }
                            },
                            ina, inb, out);
                }

                void inline matrix_multiply_u8(
                        BIIterator &ina, BIIterator &inb, BIIterator &out, int width_b, const BITensorInfo &out_info,
                        const BIWindow &window) {
                    const auto width_out = static_cast<int>(out_info.dimension(0));
                    const auto height_out = static_cast<int>(out_info.dimension(1));
                    const size_t out_stride = out_info.strides_in_bytes()[1] / out_info.element_size();
                    execute_window_loop(
                            window,
                            [&](const BICoordinates &id) {
                                const uint8_t *mtx_a0 = ina.ptr();
                                const uint8_t *mtx_b0 = inb.ptr();

                                // Note: Since the input are all positives, we can use uint32_t
                                // Accumulators for the block 0
                                uint32x4x4_t c0 = {{vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)}};

                                // Accumulators for the block 1
                                uint32x4x4_t c1 = {{vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)}};

                                // Accumulators for the block 2
                                uint32x4x4_t c2 = {{vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)}};

                                // Accumulators for the block 3
                                uint32x4x4_t c3 = {{vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)}};

                                for (int k = 0; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16) {
                                    const uint8x8_t a00_u8 = vld1_u8(mtx_a0);
                                    const uint8x16_t b00_u8 = vld1q_u8(mtx_b0);

                                    // Convert a00_u8 to uint16_t and get the lower part
                                    const uint16x4_t a00_u16 = vget_low_u16(vmovl_u8(a00_u8));

                                    // Convert b00_s8 to uint16_t
                                    const uint16x4x4_t b00_u16 = {
                                            {vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                                             vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                                             vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                                             vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))}};

                                    // 4x4 block 0
                                    c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16, 0);
                                    c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16, 0);
                                    c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16, 0);
                                    c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16, 0);

                                    // 4x4 block 1
                                    c1.val[0] = vmlal_lane_u16(c1.val[0], b00_u16.val[0], a00_u16, 1);
                                    c1.val[1] = vmlal_lane_u16(c1.val[1], b00_u16.val[1], a00_u16, 1);
                                    c1.val[2] = vmlal_lane_u16(c1.val[2], b00_u16.val[2], a00_u16, 1);
                                    c1.val[3] = vmlal_lane_u16(c1.val[3], b00_u16.val[3], a00_u16, 1);

                                    // 4x4 block 2
                                    c2.val[0] = vmlal_lane_u16(c2.val[0], b00_u16.val[0], a00_u16, 2);
                                    c2.val[1] = vmlal_lane_u16(c2.val[1], b00_u16.val[1], a00_u16, 2);
                                    c2.val[2] = vmlal_lane_u16(c2.val[2], b00_u16.val[2], a00_u16, 2);
                                    c2.val[3] = vmlal_lane_u16(c2.val[3], b00_u16.val[3], a00_u16, 2);

                                    // 4x4 block 3
                                    c3.val[0] = vmlal_lane_u16(c3.val[0], b00_u16.val[0], a00_u16, 3);
                                    c3.val[1] = vmlal_lane_u16(c3.val[1], b00_u16.val[1], a00_u16, 3);
                                    c3.val[2] = vmlal_lane_u16(c3.val[2], b00_u16.val[2], a00_u16, 3);
                                    c3.val[3] = vmlal_lane_u16(c3.val[3], b00_u16.val[3], a00_u16, 3);
                                }

                                auto mtx_out = reinterpret_cast<int32_t *>(out.ptr());

                                if (id.y() < height_out && id.x() < (width_out - 16)) {
                                    vst1q_s32(mtx_out + 0 * out_stride + 0, vreinterpretq_s32_u32(c0.val[0]));
                                    vst1q_s32(mtx_out + 0 * out_stride + 4, vreinterpretq_s32_u32(c0.val[1]));
                                    vst1q_s32(mtx_out + 0 * out_stride + 8, vreinterpretq_s32_u32(c0.val[2]));
                                    vst1q_s32(mtx_out + 0 * out_stride + 12, vreinterpretq_s32_u32(c0.val[3]));
                                    if (id.y() + 1 < height_out) {
                                        vst1q_s32(mtx_out + 1 * out_stride + 0, vreinterpretq_s32_u32(c1.val[0]));
                                        vst1q_s32(mtx_out + 1 * out_stride + 4, vreinterpretq_s32_u32(c1.val[1]));
                                        vst1q_s32(mtx_out + 1 * out_stride + 8, vreinterpretq_s32_u32(c1.val[2]));
                                        vst1q_s32(mtx_out + 1 * out_stride + 12, vreinterpretq_s32_u32(c1.val[3]));
                                        if (id.y() + 2 < height_out) {
                                            vst1q_s32(mtx_out + 2 * out_stride + 0, vreinterpretq_s32_u32(c2.val[0]));
                                            vst1q_s32(mtx_out + 2 * out_stride + 4, vreinterpretq_s32_u32(c2.val[1]));
                                            vst1q_s32(mtx_out + 2 * out_stride + 8, vreinterpretq_s32_u32(c2.val[2]));
                                            vst1q_s32(mtx_out + 2 * out_stride + 12, vreinterpretq_s32_u32(c2.val[3]));
                                            if (id.y() + 3 < height_out) {
                                                vst1q_s32(mtx_out + 3 * out_stride + 0,
                                                          vreinterpretq_s32_u32(c3.val[0]));
                                                vst1q_s32(mtx_out + 3 * out_stride + 4,
                                                          vreinterpretq_s32_u32(c3.val[1]));
                                                vst1q_s32(mtx_out + 3 * out_stride + 8,
                                                          vreinterpretq_s32_u32(c3.val[2]));
                                                vst1q_s32(mtx_out + 3 * out_stride + 12,
                                                          vreinterpretq_s32_u32(c3.val[3]));
                                            }
                                        }
                                    }
                                } else {
                                    const auto left_over_value = width_out - id.x();
                                    auto left_over = left_over_value;
                                    for (auto k = 0; k < 4 && left_over; ++k) {
                                        for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                            *(mtx_out + k * 4 + j) = c0.val[k][j];
                                        }
                                    }
                                    if (id.y() + 1 < height_out) {
                                        left_over = left_over_value;
                                        for (auto k = 0; k < 4 && left_over; ++k) {
                                            for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                                *(mtx_out + out_stride + k * 4 + j) = c1.val[k][j];
                                            }
                                        }
                                        if (id.y() + 2 < height_out) {
                                            left_over = left_over_value;
                                            for (auto k = 0; k < 4 && left_over; ++k) {
                                                for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                                    *(mtx_out + out_stride * 2 + k * 4 + j) = c2.val[k][j];
                                                }
                                            }
                                            if (id.y() + 3 < height_out) {
                                                left_over = left_over_value;
                                                for (auto k = 0; k < 4 && left_over; ++k) {
                                                    for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                                        *(mtx_out + out_stride * 3 + k * 4 + j) = c3.val[k][j];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            ina, inb, out);
                }

                void inline matrix_multiply_s8(
                        BIIterator &ina, BIIterator &inb, BIIterator &out, int width_b, const BITensorInfo &out_info,
                        const BIWindow &window) {
                    const auto width_out = static_cast<int>(out_info.dimension(0));
                    const auto height_out = static_cast<int>(out_info.dimension(1));
                    const size_t out_stride = out_info.strides_in_bytes()[1] / out_info.element_size();
                    // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with CpuGemmInterleave4x4 and CpuGemmTranspose1xW
                    // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
                    // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
                    execute_window_loop(
                            window,
                            [&](const BICoordinates &id) {
                                auto *mtx_a0 = reinterpret_cast<const int8_t *>(ina.ptr());
                                auto *mtx_b0 = reinterpret_cast<const int8_t *>(inb.ptr());

                                // Note: Since the input are all positives, we can use uint32_t
                                // Accumulators for the block 0
                                int32x4x4_t c0 = {{vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)}};

                                // Accumulators for the block 1
                                int32x4x4_t c1 = {{vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)}};

                                // Accumulators for the block 2
                                int32x4x4_t c2 = {{vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)}};

                                // Accumulators for the block 3
                                int32x4x4_t c3 = {{vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)}};

                                for (int k = 0; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16) {
                                    const int8x8_t a00_s8 = vld1_s8(mtx_a0);
                                    const int8x16_t b00_s8 = vld1q_s8(mtx_b0);

                                    // Convert a00_s8 to uint16_t and get the lower part
                                    const int16x4_t a00_s16 = vget_low_s16(vmovl_s8(a00_s8));

                                    // Convert b00_s8 to int16_t
                                    const int16x4x4_t b00_s16 = {
                                            {vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                                             vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                                             vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                                             vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))}};

                                    // 4x4 block 0
                                    c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16, 0);
                                    c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16, 0);
                                    c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16, 0);
                                    c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16, 0);

                                    // 4x4 block 1
                                    c1.val[0] = vmlal_lane_s16(c1.val[0], b00_s16.val[0], a00_s16, 1);
                                    c1.val[1] = vmlal_lane_s16(c1.val[1], b00_s16.val[1], a00_s16, 1);
                                    c1.val[2] = vmlal_lane_s16(c1.val[2], b00_s16.val[2], a00_s16, 1);
                                    c1.val[3] = vmlal_lane_s16(c1.val[3], b00_s16.val[3], a00_s16, 1);

                                    // 4x4 block 2
                                    c2.val[0] = vmlal_lane_s16(c2.val[0], b00_s16.val[0], a00_s16, 2);
                                    c2.val[1] = vmlal_lane_s16(c2.val[1], b00_s16.val[1], a00_s16, 2);
                                    c2.val[2] = vmlal_lane_s16(c2.val[2], b00_s16.val[2], a00_s16, 2);
                                    c2.val[3] = vmlal_lane_s16(c2.val[3], b00_s16.val[3], a00_s16, 2);

                                    // 4x4 block 3
                                    c3.val[0] = vmlal_lane_s16(c3.val[0], b00_s16.val[0], a00_s16, 3);
                                    c3.val[1] = vmlal_lane_s16(c3.val[1], b00_s16.val[1], a00_s16, 3);
                                    c3.val[2] = vmlal_lane_s16(c3.val[2], b00_s16.val[2], a00_s16, 3);
                                    c3.val[3] = vmlal_lane_s16(c3.val[3], b00_s16.val[3], a00_s16, 3);
                                }
                                auto mtx_out = reinterpret_cast<int32_t *>(out.ptr());
                                if (id.y() < height_out && id.x() < (width_out - 16)) {
                                    vst1q_s32(mtx_out + 0 * out_stride + 0, c0.val[0]);
                                    vst1q_s32(mtx_out + 0 * out_stride + 4, c0.val[1]);
                                    vst1q_s32(mtx_out + 0 * out_stride + 8, c0.val[2]);
                                    vst1q_s32(mtx_out + 0 * out_stride + 12, c0.val[3]);
                                    if (id.y() + 1 < height_out) {
                                        vst1q_s32(mtx_out + 1 * out_stride + 0, c1.val[0]);
                                        vst1q_s32(mtx_out + 1 * out_stride + 4, c1.val[1]);
                                        vst1q_s32(mtx_out + 1 * out_stride + 8, c1.val[2]);
                                        vst1q_s32(mtx_out + 1 * out_stride + 12, c1.val[3]);
                                        if (id.y() + 2 < height_out) {
                                            vst1q_s32(mtx_out + 2 * out_stride + 0, c2.val[0]);
                                            vst1q_s32(mtx_out + 2 * out_stride + 4, c2.val[1]);
                                            vst1q_s32(mtx_out + 2 * out_stride + 8, c2.val[2]);
                                            vst1q_s32(mtx_out + 2 * out_stride + 12, c2.val[3]);
                                            if (id.y() + 3 < height_out) {
                                                vst1q_s32(mtx_out + 3 * out_stride + 0, c3.val[0]);
                                                vst1q_s32(mtx_out + 3 * out_stride + 4, c3.val[1]);
                                                vst1q_s32(mtx_out + 3 * out_stride + 8, c3.val[2]);
                                                vst1q_s32(mtx_out + 3 * out_stride + 12, c3.val[3]);
                                            }
                                        }
                                    }
                                } else if (id.y() < height_out) {
                                    const auto left_over_value = width_out - id.x();
                                    auto left_over = left_over_value;
                                    for (auto k = 0; k < 4 && left_over; ++k) {
                                        for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                            *(mtx_out + k * 4 + j) = c0.val[k][j];
                                        }
                                    }
                                    if (id.y() + 1 < height_out) {
                                        left_over = left_over_value;
                                        for (auto k = 0; k < 4 && left_over; ++k) {
                                            for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                                *(mtx_out + out_stride + k * 4 + j) = c1.val[k][j];
                                            }
                                        }
                                        if (id.y() + 2 < height_out) {
                                            left_over = left_over_value;
                                            for (auto k = 0; k < 4 && left_over; ++k) {
                                                for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                                    *(mtx_out + out_stride * 2 + k * 4 + j) = c2.val[k][j];
                                                }
                                            }
                                            if (id.y() + 3 < height_out) {
                                                left_over = left_over_value;
                                                for (auto k = 0; k < 4 && left_over; ++k) {
                                                    for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                                        *(mtx_out + out_stride * 3 + k * 4 + j) = c3.val[k][j];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            ina, inb, out);
                }

                BIStatus
                validate_arguments(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst) {
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::S8, BIDataType::U8);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::QSYMM8,
                                                                        BIDataType::QSYMM8_PER_CHANNEL,
                                                                        BIDataType::S8,
                                                                        BIDataType::U8);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::S32);

                    BI_COMPUTE_RETURN_ERROR_ON_MSG(src0->data_type() == BIDataType::QASYMM8_SIGNED &&
                                                   src1->data_type() == BIDataType::QASYMM8,
                                                   "QASYMM8_SIGNED input with QASYMM8 weights not supported");

                    BITensorShape in0_shape = src0->tensor_shape();
                    BITensorShape in1_shape = src1->tensor_shape();
                    BITensorShape out_shape = dst->tensor_shape();

                    // Check vector-by-matrix case
                    if (out_shape[1] == 1) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(in0_shape[0] != in1_shape[1],
                                                       "The number of input0's columns must be equal to input1's rows");
                    } else {
                        in0_shape.collapse(2);
                        in1_shape.collapse(2);
                        out_shape.collapse(2);

                        BI_COMPUTE_RETURN_ERROR_ON_MSG(in0_shape[2] != out_shape[2],
                                                       "Output tensor must have the same number of batches of input0 tensor");
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                in1_shape[2] != 1 && in0_shape[2] != in1_shape[2],
                                "Input1 tensor must have the same number of batches of input0 or the number of batches must be set to 1");
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(in1_shape[0] % 16, "Input1's width must be a multiple of 16");
                    }

                    return BIStatus{};
                }
            } // namespace

            void BICpuGemmLowpMatrixMultiplyKernel::configure(const BatmanInfer::BIITensorInfo *src0,
                                                              const BatmanInfer::BIITensorInfo *src1,
                                                              BatmanInfer::BIITensorInfo *dst) {
                BI_COMPUTE_UNUSED(src0);
                BI_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, dst));

                BITensorShape in1_shape = src1->tensor_shape();
                in1_shape.collapse(2);

                _slide_matrix_b = in1_shape[2] != 1;

                constexpr unsigned int num_elems_processed_per_iteration_x = 16;
                constexpr unsigned int num_elems_processed_per_iteration_y = 4;

                BIWindow win;
                // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication
                if ((dst->dimension(1) == 1)) {
                    // Configure kernel window
                    win = calculate_max_window(*dst, BISteps(num_elems_processed_per_iteration_x));
                } else {
                    win =
                            calculate_max_window(*dst, BISteps(num_elems_processed_per_iteration_x,
                                                               num_elems_processed_per_iteration_y));
                }

                BIICpuKernel::configure(win);
            }

            BIStatus
            BICpuGemmLowpMatrixMultiplyKernel::validate(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                        const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src0, src1, dst));
                return BIStatus{};
            }

            void BICpuGemmLowpMatrixMultiplyKernel::run_op(BIITensorPack &tensors, const BIWindow &window,
                                                           const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                auto src0 = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                auto src1 = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication path
                if ((dst->info()->dimension(1) == 1)) {
                    const auto width_matrix_a = static_cast<int>(src0->info()->dimension(0));
                    const auto width_matrix_b = static_cast<int>(src1->info()->dimension(0));
                    const auto width_out = static_cast<int>(dst->info()->dimension(0));
                    const auto in_b_stride =
                            static_cast<int>(src1->info()->strides_in_bytes()[1] /
                                             data_size_from_type(src1->info()->data_type()));

                    // The implementation computes 16 elements per iteration
                    const int window_start_x = 16 * info.thread_id;
                    const int window_step_x = 16 * info.num_threads;
                    // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
                    const int window_end_x =
                            ceil_to_multiples(width_matrix_b - window_start_x, window_step_x) + window_start_x;

                    BIWindow win_out(window);
                    win_out.set(BIWindow::DimX, BIWindow::BIDimension(window_start_x, window_end_x, window_step_x));
                    win_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

                    BIWindow win_a(window);
                    win_a.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    win_a.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                    BIWindow win_b;
                    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
                    // This scenario can happen when the matrix multiplication is used to perform a convolution operation
                    if (src1->info()->num_dimensions() >= 3) {
                        win_b = window;
                    }
                    win_b.set(BIWindow::DimX, BIWindow::BIDimension(window_start_x, window_end_x, window_step_x));
                    win_b.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

                    BIIterator ina(src0, win_a);
                    BIIterator inb(src1, win_b);
                    BIIterator out(dst, win_out);

                    switch (src0->info()->data_type()) {
                        case BIDataType::S8:
                        case BIDataType::QASYMM8_SIGNED: {
                            vector_matrix_multiply_s8(ina, inb, out, width_matrix_a, width_matrix_b, width_out,
                                                      in_b_stride,
                                                      window);
                            break;
                        }
                        case BIDataType::U8:
                        case BIDataType::QASYMM8: {
                            vector_matrix_multiply_u8(ina, inb, out, width_matrix_a, width_matrix_b, width_out,
                                                      in_b_stride,
                                                      window);
                            break;
                        }
                        default: {
                            BI_COMPUTE_ERROR("Not supported");
                            break;
                        }
                    }
                } else {
                    const size_t in_b_stride = src1->info()->strides_in_bytes()[1];
                    const int width_b = src1->info()->dimension(0);

                    // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the output matrix
                    BIWindow win_a(window);
                    win_a.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    win_a.set(BIWindow::DimY, BIWindow::BIDimension(window.y().start() / 4, window.y().end() / 4, 1));

                    // Set step_x and step_y for matrix B. Scale by a factor of 16 the X range as the input transposed matrix A has 16 times less the columns of the output matrix
                    BIWindow win_b;
                    // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
                    // This scenario can happen when the matrix multiplication is used to perform a convolution operation
                    if (_slide_matrix_b) {
                        win_b = window;
                    }
                    win_b.set(BIWindow::DimX,
                              BIWindow::BIDimension(window.x().start() / 16, window.x().end() / 16, in_b_stride));
                    win_b.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                    // The step x and step y for the output matrix has been already set using in configure()
                    BIIterator ina(src0, win_a);
                    BIIterator inb(src1, win_b);
                    BIIterator out(dst, window);

                    switch (src0->info()->data_type()) {
                        case BIDataType::S8:
                        case BIDataType::QASYMM8_SIGNED: {
                            matrix_multiply_s8(ina, inb, out, width_b, *dst->info(), window);
                            break;
                        }
                        case BIDataType::U8:
                        case BIDataType::QASYMM8: {
                            matrix_multiply_u8(ina, inb, out, width_b, *dst->info(), window);
                            break;
                        }
                        default: {
                            BI_COMPUTE_ERROR("Not supported");
                            break;
                        }
                    }
                }
            }

            const char *BICpuGemmLowpMatrixMultiplyKernel::name() const {
                return "BICpuGemmLowpMatrixMultiplyKernel";
            }
        } // namespace kernel
    }
}