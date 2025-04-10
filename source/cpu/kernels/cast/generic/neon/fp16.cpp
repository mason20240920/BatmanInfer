//
// Created by holynova on 25-4-8.
//

#include "data/core/cpp/cpp_types.hpp"
#include "data/core/bi_tensor_info.hpp"
#include "cpu/kernels/cast/list.h"
#include "cpu/kernels/bi_cpu_cast_kernel.h"
#include "support/bi_saturate_cast.hpp"

#include "neon/neon_defines.h"

namespace BatmanInfer {

namespace cpu {

    void neon_qasymm8_signed_to_fp16_cast(
        const BIITensor *_src, BIITensor *_dst, const ThreadInfo &info, BIConvertPolicy _policy, const BIWindow &window)
    {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_UNUSED(_policy);

        const auto    window_start_x = static_cast<int>(window.x().start());
        const auto    window_end_x   = static_cast<int>(window.x().end());
        constexpr int window_step_x  = 16;

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
        BI_COMPUTE_ERROR_ON(_src == _dst);

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

        BIWindow win{window};
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator src(_src, win);
        BIIterator dst(_dst, win);
        execute_window_loop(
            win,
            [&](const BICoordinates &)
            {
                const auto src_ptr = reinterpret_cast<const int8_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());
                int        x       = window_start_x;

                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const int8x16_t texels_s8 = vld1q_s8(src_ptr + x);

                    const int16x8x2_t texels = {{vmovl_s8(vget_low_s8(texels_s8)), vmovl_s8(vget_high_s8(texels_s8))}};
                    vst1q_f16(dst_ptr + x, vcvtq_f16_s16(texels.val[0]));
                    vst1q_f16(dst_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
                }
            },
            src, dst);
    }

    void neon_s32_to_fp16_cast(
    const BIITensor *_src, BIITensor *_dst, const ThreadInfo &info, BIConvertPolicy _policy, const BIWindow &window)
    {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_UNUSED(_policy);

        const auto    window_start_x = static_cast<int>(window.x().start());
        const auto    window_end_x   = static_cast<int>(window.x().end());
        constexpr int window_step_x  = 16;

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
        BI_COMPUTE_ERROR_ON(_src == _dst);

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

        BIWindow win{window};
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator src(_src, win);
        BIIterator dst(_dst, win);

        execute_window_loop(
            win,
            [&](const BICoordinates &)
            {
                const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());

                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const float32x4x4_t texels = {
                        {vcvtq_f32_s32(vld1q_s32(src_ptr + x)), vcvtq_f32_s32(vld1q_s32(src_ptr + x + 4)),
                         vcvtq_f32_s32(vld1q_s32(src_ptr + x + 8)), vcvtq_f32_s32(vld1q_s32(src_ptr + x + 12))}};

                    vst1q_f16(dst_ptr + x, vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
                    vst1q_f16(dst_ptr + x + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
                }
            },
            src, dst);
    }

    void neon_fp32_to_fp16_cast(
    const BIITensor *_src, BIITensor *_dst, const ThreadInfo &info, BIConvertPolicy _policy, const BIWindow &window)
    {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_UNUSED(_policy);

        const auto    window_start_x = static_cast<int>(window.x().start());
        const auto    window_end_x   = static_cast<int>(window.x().end());
        constexpr int window_step_x  = 16;

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
        BI_COMPUTE_ERROR_ON(_src == _dst);

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

        BIWindow win{window};
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator src(_src, win);
        BIIterator dst(_dst, win);

        execute_window_loop(
            win,
            [&](const BICoordinates &)
            {
                const auto src_ptr = reinterpret_cast<const float *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());

                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const float32x4x4_t texels = {{vld1q_f32(src_ptr + x), vld1q_f32(src_ptr + x + 4),
                                                   vld1q_f32(src_ptr + x + 8), vld1q_f32(src_ptr + x + 12)}};

                    vst1q_f16(dst_ptr + x, vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
                    vst1q_f16(dst_ptr + x + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
                }
            },
            src, dst);
    }

    void neon_fp16_to_other_dt_cast(
    const BIITensor *_src, BIITensor *_dst, const ThreadInfo &info, BIConvertPolicy _policy, const BIWindow &window)
    {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_UNUSED(_policy);

        const auto    window_start_x = static_cast<int>(window.x().start());
        const auto    window_end_x   = static_cast<int>(window.x().end());
        constexpr int window_step_x  = 16;

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
        BI_COMPUTE_ERROR_ON(_src == _dst);

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

        BIWindow win{window};
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator src(_src, win);
        BIIterator dst(_dst, win);
        switch (_dst->info()->data_type())
        {
            case BIDataType::QASYMM8_SIGNED:
            {
                /* Down-conversion F16 -> QASYMM8_SIGNED (Always saturating) */
                execute_window_loop(
                    win,
                    [&](const BICoordinates &)
                    {
                        const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                        const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels = {{
                                vld1q_f16(src_ptr + x),
                                vld1q_f16(src_ptr + x + 8),
                            }};

                            vst1q_s8(dst_ptr + x, vcombine_s8(vqmovn_s16(vcvtq_s16_f16(texels.val[0])),
                                                              vqmovn_s16(vcvtq_s16_f16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            *(dst_ptr + x) = utils::cast::saturate_static_cast<int8_t>(*(src_ptr + x));
                        }
                    },
                    src, dst);
                break;
            }
            case BIDataType::QASYMM8:
            case BIDataType::U8:
            {
                /* Down-conversion F16 -> QASYMM8/U8 (Always saturating) */
                execute_window_loop(
                    win,
                    [&](const BICoordinates &)
                    {
                        const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                        const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels = {{
                                vld1q_f16(src_ptr + x),
                                vld1q_f16(src_ptr + x + 8),
                            }};

                            vst1q_u8(dst_ptr + x, vcombine_u8(vqmovun_s16(vcvtq_s16_f16(texels.val[0])),
                                                              vqmovun_s16(vcvtq_s16_f16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            *(dst_ptr + x) = utils::cast::saturate_static_cast<uint8_t>(*(src_ptr + x));
                        }
                    },
                    src, dst);
                break;
            }
            case BIDataType::F32:
            {
                /* Up-conversion F16 -> F32 */
                execute_window_loop(
                    win,
                    [&](const BICoordinates &)
                    {
                        const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                        const auto dst_ptr = reinterpret_cast<float *>(dst.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels = {{vld1q_f16(src_ptr + x), vld1q_f16(src_ptr + x + 8)}};
                            vst1q_f32(dst_ptr + x, vcvt_f32_f16(vget_low_f16(texels.val[0])));
                            vst1q_f32(dst_ptr + x + 4, vcvt_f32_f16(vget_high_f16(texels.val[0])));
                            vst1q_f32(dst_ptr + x + 8, vcvt_f32_f16(vget_low_f16(texels.val[1])));
                            vst1q_f32(dst_ptr + x + 12, vcvt_f32_f16(vget_high_f16(texels.val[1])));
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            *(dst_ptr + x) = static_cast<float>(*(src_ptr + x));
                        }
                    },
                    src, dst);
                break;
            }
            case BIDataType::S32:
            {
                /* Up-conversion F16 -> S32 */
                execute_window_loop(
                    win,
                    [&](const BICoordinates &)
                    {
                        const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                        const auto dst_ptr = reinterpret_cast<int32_t *>(dst.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels = {{vld1q_f16(src_ptr + x), vld1q_f16(src_ptr + x + 8)}};

                            vst1q_s32(dst_ptr + x, vcvtq_s32_f32(vcvt_f32_f16(vget_low_f16(texels.val[0]))));
                            vst1q_s32(dst_ptr + x + 4, vcvtq_s32_f32(vcvt_f32_f16(vget_high_f16(texels.val[0]))));
                            vst1q_s32(dst_ptr + x + 8, vcvtq_s32_f32(vcvt_f32_f16(vget_low_f16(texels.val[1]))));
                            vst1q_s32(dst_ptr + x + 12, vcvtq_s32_f32(vcvt_f32_f16(vget_high_f16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x)
                        {
                            *(dst_ptr + x) = static_cast<int32_t>(*(src_ptr + x));
                        }
                    },
                    src, dst);
                break;
            }
            default:
                BI_COMPUTE_ERROR("dst data type not supported");
        }
    }

    void neon_u8_to_fp16_cast(
    const BIITensor *_src, BIITensor *_dst, const ThreadInfo &info, BIConvertPolicy _policy, const BIWindow &window)
    {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_UNUSED(_policy);

        const auto    window_start_x = static_cast<int>(window.x().start());
        const auto    window_end_x   = static_cast<int>(window.x().end());
        constexpr int window_step_x  = 16;

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
        BI_COMPUTE_ERROR_ON(_src == _dst);

        BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

        BIWindow win{window};
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator src(_src, win);
        BIIterator dst(_dst, win);
        /* Up-conversion U8 -> F16 */
        execute_window_loop(
            win,
            [&](const BICoordinates &)
            {
                const auto src_ptr = reinterpret_cast<const uint8_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());

                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const uint8x16_t texels_u8 = vld1q_u8(src_ptr + x);

                    const int16x8x2_t texels = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))),
                                                 vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8)))}};
                    vst1q_f16(dst_ptr + x, vcvtq_f16_s16(texels.val[0]));
                    vst1q_f16(dst_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
                }
            },
            src, dst);
        return;
    }

} // namespace cpu

} // namespace BatmanInfer
