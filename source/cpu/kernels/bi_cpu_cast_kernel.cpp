//
// Created by holynova on 25-4-8.
//

#include "cpu/kernels/bi_cpu_cast_kernel.h"

#include "data/core/bi_error.h"
#include "data/core/bi_helpers.hpp"
#include "data/core/bi_i_tensor.hpp"
#include "data/core/bi_tensor_info.hpp"
#include "data/core/bi_vlidate.hpp"
#include "common/bi_registers.hpp"
#include "data/core/cpp/bi_cpp_validate.hpp"
#include "data/core/helpers/bi_auto_configuration.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"
#include "data/core/neon/bi_neon_fixed_point.h"
#include "data/core/neon/bi_neon_math.hpp"
#include "data/core/neon/wrapper/wrapper.hpp"
#include "cpu/kernels/cast/list.h"
#include "support/bi_saturate_cast.hpp"

namespace BatmanInfer {

namespace cpu {

namespace kernels {

namespace {

    static const std::vector<BICpuCastKernel::CastKernel> available_kernels = {
        {"neon_qs8_cast",
         [](const BICastDataTypeISASelectorData &data)
         { return data.src_dt == BIDataType::QASYMM8_SIGNED && data.dst_dt == BIDataType::F16 && data.isa.fp16; },
         REGISTER_FP16_NEON(BatmanInfer::cpu::neon_qasymm8_signed_to_fp16_cast)},
        {"neon_qu8_cast",
         [](const BICastDataTypeISASelectorData &data)
         { return data.src_dt == BIDataType::QASYMM8 && data.dst_dt == BIDataType::F16 && data.isa.fp16; },
         REGISTER_FP16_NEON(BatmanInfer::cpu::neon_u8_to_fp16_cast)},
        {"neon_u8_cast",
         [](const BICastDataTypeISASelectorData &data)
         { return data.src_dt == BIDataType::U8 && data.dst_dt == BIDataType::F16 && data.isa.fp16; },
         REGISTER_FP16_NEON(BatmanInfer::cpu::neon_u8_to_fp16_cast)},
        {"neon_fp16_cast",
         [](const BICastDataTypeISASelectorData &data) { return data.src_dt == BIDataType::F16 && data.isa.fp16; },
         REGISTER_FP16_NEON(BatmanInfer::cpu::neon_fp16_to_other_dt_cast)},
        {"neon_fp32_to_fp16_cast",
         [](const BICastDataTypeISASelectorData &data)
         { return data.src_dt == BIDataType::F32 && data.dst_dt == BIDataType::F16 && data.isa.fp16; },
         REGISTER_FP16_NEON(BatmanInfer::cpu::neon_fp32_to_fp16_cast)},
        {"neon_s32_cast",
         [](const BICastDataTypeISASelectorData &data)
         { return data.src_dt == BIDataType::S32 && data.dst_dt == BIDataType::F16 && data.isa.fp16; },
         REGISTER_FP16_NEON(BatmanInfer::cpu::neon_s32_to_fp16_cast)},
    };

    BIStatus validate_arguments(const BIITensorInfo *src, const BIITensorInfo *dst, BIConvertPolicy policy)
    {
        BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
        BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(dst);
        BI_COMPUTE_UNUSED(policy);
        BI_COMPUTE_RETURN_ERROR_ON(src == dst);
#ifdef __aarch64__
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8_SIGNED, BIDataType::QASYMM8,
                                                            BIDataType::U8, BIDataType::S16, BIDataType::U16,
                                                            BIDataType::F16, BIDataType::F32, BIDataType::S32,
                                                            BIDataType::S64, BIDataType::U64);

        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::QASYMM8_SIGNED, BIDataType::QASYMM8,
                                                            BIDataType::U8, BIDataType::S16, BIDataType::U16,
                                                            BIDataType::F16, BIDataType::U32, BIDataType::S32,
                                                            BIDataType::F32, BIDataType::S64);

#else  // __aarch64__
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8_SIGNED, BIDataType::QASYMM8,
                                                            BIDataType::U8, BIDataType::S16, BIDataType::U16,
                                                            BIDataType::F16, BIDataType::F32, BIDataType::S32);

        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::QASYMM8_SIGNED, BIDataType::QASYMM8,
                                                            BIDataType::U8, BIDataType::S16, BIDataType::U16,
                                                            BIDataType::F16, BIDataType::U32, BIDataType::S32,
                                                            BIDataType::F32);
#endif // __aarch64__

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::QASYMM8_SIGNED &&
                                       (dst->data_type() != BIDataType::S16 && dst->data_type() != BIDataType::S32 &&
                                       dst->data_type() != BIDataType::F16 && dst->data_type() != BIDataType::F32),
                                       "Only data_types supported [in] QASYMM8 -> [out] U16, S16, S32, F16, F32");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::QASYMM8 &&
                                       (dst->data_type() != BIDataType::S16 && dst->data_type() != BIDataType::U16 &&
                                       dst->data_type() != BIDataType::S32 && dst->data_type() != BIDataType::F16 &&
                                       dst->data_type() != BIDataType::F32),
                                       "Only data_types supported [in] QASYMM8 -> [out] U16, S16, S32, F16, F32");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::U8 &&
                                       (dst->data_type() != BIDataType::S16 && dst->data_type() != BIDataType::U16 &&
                                       dst->data_type() != BIDataType::S32 && dst->data_type() != BIDataType::F16 &&
                                       dst->data_type() != BIDataType::F32),
                                       "Only data_types supported [in] U8 -> [out] U16, S16, S32, F16, F32");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::U16 &&
                                       (dst->data_type() != BIDataType::U8 && dst->data_type() != BIDataType::U32),
                                       "Only data_types supported [in] U16 ->  [out] U8, U32");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::S16 &&
                                       (dst->data_type() != BIDataType::QASYMM8_SIGNED &&
                                       dst->data_type() != BIDataType::U8 && dst->data_type() != BIDataType::S32),
                                       "Only data_types supported [in] S16 ->  [out] U8, S32");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::F16 &&
                                       (dst->data_type() != BIDataType::QASYMM8_SIGNED &&
                                       dst->data_type() != BIDataType::QASYMM8 && dst->data_type() != BIDataType::U8 &&
                                       dst->data_type() != BIDataType::F32 && dst->data_type() != BIDataType::S32),
                                       "Only data_types supported [in] F16 ->  [out] QASYMM8, F32, S32, U8");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::F32 &&
                                       (dst->data_type() != BIDataType::QASYMM8_SIGNED &&
                                       dst->data_type() != BIDataType::QASYMM8 && dst->data_type() != BIDataType::F16 &&
                                       dst->data_type() != BIDataType::S32 && dst->data_type() != BIDataType::U8),
                                       "Only data_types supported [in] F32 ->  [out] QASYMM8, F16, S32, U8");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::S32 &&
                                       (dst->data_type() != BIDataType::QASYMM8_SIGNED &&
                                       dst->data_type() != BIDataType::QASYMM8 && dst->data_type() != BIDataType::F16 &&
                                       dst->data_type() != BIDataType::F32 && dst->data_type() != BIDataType::U8 &&
                                       dst->data_type() != BIDataType::S64),
                                       "Only data_types supported [in] S32 ->  [out] QASYMM8, F16, F32, U8, S64");
#ifdef __aarch64__
        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::S64 && dst->data_type() != BIDataType::F32,
                                       "Only data_types supported [in] S64 ->  [out] F32");

        BI_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == BIDataType::U64 && dst->data_type() != BIDataType::F32,
                                       "Only data_types supported [in] U64 ->  [out] F32");
#endif // __aarch64__

        // Validate in case of configured dst
        if (dst->total_size() > 0)
        {
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
        }

        return BIStatus{};
    }

} // namespace

void BICpuCastKernel::configure(const BIITensorInfo *src, BIITensorInfo *dst, BIConvertPolicy policy)
{
    BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Auto initialize dst shape if not initialized (We can only auto-configure the shape, datatype must be given)
    set_shape_if_empty(*dst, src->tensor_shape());

    _policy = policy;

    BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, policy));

    // Configure kernel window
    BIWindow win = calculate_max_window(*src, BISteps());

    BIICPPKernel::configure(win);
}

BIStatus BICpuCastKernel::validate(const BIITensorInfo *src, const BIITensorInfo *dst, BIConvertPolicy policy)
{
    BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, policy));
    return BIStatus{};
}

#ifdef __aarch64__
namespace {

    template <typename T1, typename T2>
    inline void internal_neon_convert(const T1 *src_ptr, T2 *dst_ptr)
    {
        BI_COMPUTE_UNUSED(src_ptr);
        BI_COMPUTE_UNUSED(dst_ptr);
    }

    template <>
    inline void internal_neon_convert<int32_t, int64_t>(const int32_t *src_ptr, int64_t *dst_ptr)
    {
        const int32x4x4_t texels = {
            {vld1q_s32(src_ptr), vld1q_s32(src_ptr + 4), vld1q_s32(src_ptr + 8), vld1q_s32(src_ptr + 12)}};
        vst1q_s64(dst_ptr, vmovl_s32(vget_low_s32(texels.val[0])));
        vst1q_s64(dst_ptr + 2, vmovl_s32(vget_high_s32(texels.val[0])));
        vst1q_s64(dst_ptr + 4, vmovl_s32(vget_low_s32(texels.val[1])));
        vst1q_s64(dst_ptr + 6, vmovl_s32(vget_high_s32(texels.val[1])));
        vst1q_s64(dst_ptr + 8, vmovl_s32(vget_low_s32(texels.val[2])));
        vst1q_s64(dst_ptr + 10, vmovl_s32(vget_high_s32(texels.val[2])));
        vst1q_s64(dst_ptr + 12, vmovl_s32(vget_low_s32(texels.val[3])));
        vst1q_s64(dst_ptr + 14, vmovl_s32(vget_high_s32(texels.val[3])));
    }

    template <>
    inline void internal_neon_convert<int64_t, float>(const int64_t *src_ptr, float *dst_ptr)
    {
        const float64x2x4_t texels0 = {{vcvtq_f64_s64(vld1q_s64(src_ptr)), vcvtq_f64_s64(vld1q_s64(src_ptr + 2)),
                                        vcvtq_f64_s64(vld1q_s64(src_ptr + 4)), vcvtq_f64_s64(vld1q_s64(src_ptr + 6))}};
        const float64x2x4_t texels1 = {{vcvtq_f64_s64(vld1q_s64(src_ptr + 8)), vcvtq_f64_s64(vld1q_s64(src_ptr + 10)),
                                        vcvtq_f64_s64(vld1q_s64(src_ptr + 12)), vcvtq_f64_s64(vld1q_s64(src_ptr + 14))}};
        const float32x4x4_t texels  = {{vcombine_f32(vcvt_f32_f64(texels0.val[0]), vcvt_f32_f64(texels0.val[1])),
                                        vcombine_f32(vcvt_f32_f64(texels0.val[2]), vcvt_f32_f64(texels0.val[3])),
                                        vcombine_f32(vcvt_f32_f64(texels1.val[0]), vcvt_f32_f64(texels1.val[1])),
                                        vcombine_f32(vcvt_f32_f64(texels1.val[2]), vcvt_f32_f64(texels1.val[3]))}};
        vst1q_f32(dst_ptr, texels.val[0]);
        vst1q_f32(dst_ptr + 4, texels.val[1]);
        vst1q_f32(dst_ptr + 8, texels.val[2]);
        vst1q_f32(dst_ptr + 12, texels.val[3]);
    }

    template <>
    inline void internal_neon_convert<uint64_t, float>(const uint64_t *src_ptr, float *dst_ptr)
    {
        const float64x2x4_t texels0 = {{vcvtq_f64_u64(vld1q_u64(src_ptr)), vcvtq_f64_u64(vld1q_u64(src_ptr + 2)),
                                        vcvtq_f64_u64(vld1q_u64(src_ptr + 4)), vcvtq_f64_u64(vld1q_u64(src_ptr + 6))}};
        const float64x2x4_t texels1 = {{vcvtq_f64_u64(vld1q_u64(src_ptr + 8)), vcvtq_f64_u64(vld1q_u64(src_ptr + 10)),
                                        vcvtq_f64_u64(vld1q_u64(src_ptr + 12)), vcvtq_f64_u64(vld1q_u64(src_ptr + 14))}};

        const float32x4x4_t texels = {{vcombine_f32(vcvt_f32_f64(texels0.val[0]), vcvt_f32_f64(texels0.val[1])),
                                       vcombine_f32(vcvt_f32_f64(texels0.val[2]), vcvt_f32_f64(texels0.val[3])),
                                       vcombine_f32(vcvt_f32_f64(texels1.val[0]), vcvt_f32_f64(texels1.val[1])),
                                       vcombine_f32(vcvt_f32_f64(texels1.val[2]), vcvt_f32_f64(texels1.val[3]))}};

        vst1q_f32(dst_ptr, texels.val[0]);
        vst1q_f32(dst_ptr + 4, texels.val[1]);
        vst1q_f32(dst_ptr + 8, texels.val[2]);
        vst1q_f32(dst_ptr + 12, texels.val[3]);
    }

    template <typename T1, typename T2>
    inline void
    convert64(BIIterator &src, BIIterator &dst, const BIWindow &win, int window_start_x, int window_end_x, int window_step_x)
    {
        execute_window_loop(
            win,
            [&](const BICoordinates &)
            {
                const auto src_ptr = reinterpret_cast<const T1 *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<T2 *>(dst.ptr());
                int        x       = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    internal_neon_convert<T1, T2>(src_ptr + x, dst_ptr + x);
                }
                for (; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<T2>(*(src_ptr + x));
                }
            },
            src, dst);
    }

} // namespace
#endif // __aarch64__

void BICpuCastKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info)
{
    BI_COMPUTE_UNUSED(info);
    BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIIKernel::window(), window);

    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());
    constexpr int window_step_x  = 16;

    const BIITensor *_src = tensors.get_const_tensor(BITensorType::ACL_SRC);
    BIITensor       *_dst = tensors.get_tensor(BITensorType::ACL_DST);
    BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    BI_COMPUTE_ERROR_ON(_src == _dst);

    BI_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    BIWindow win{window};
    win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

    BIIterator src(_src, win);
    BIIterator dst(_dst, win);

    /*ukernel runs only when using fp16, so we validate it isn't a nullptr only before using it */
    const auto *uk = BICpuCastKernel::get_implementation(
        BICastDataTypeISASelectorData{_src->info()->data_type(), _dst->info()->data_type(), CPUInfo::get().get_isa()});

    switch (_src->info()->data_type())
    {
#ifdef __aarch64__
        case BIDataType::U64:
        {
            switch (_dst->info()->data_type())
            {
                case BIDataType::F32:
                {
                    convert64<uint64_t, float>(src, dst, win, window_start_x, window_end_x, window_step_x);
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        }
        case BIDataType::S64:
        {
            switch (_dst->info()->data_type())
            {
                case BIDataType::F32:
                {
                    convert64<int64_t, float>(src, dst, win, window_start_x, window_end_x, window_step_x);
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        }
#endif // __aarch64__

        case BIDataType::QASYMM8_SIGNED:
        {
            switch (_dst->info()->data_type())
            {
                case BIDataType::S16:
                {
                    /* Up-conversion QASYMM8_SIGNED -> S16 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const int8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int16_t *>(dst.ptr());
                            int        x       = window_start_x;

                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int8x16_t texels_s8 = vld1q_s8(src_ptr + x);

                                const int16x8x2_t texels = {
                                    {vmovl_s8(vget_low_s8(texels_s8)), vmovl_s8(vget_high_s8(texels_s8))}};

                                vst1q_s16(dst_ptr + x, texels.val[0]);
                                vst1q_s16(dst_ptr + x + 8, texels.val[1]);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                *(dst_ptr + x) = static_cast<int16_t>(*(src_ptr + x));
                            }
                        },
                        src, dst);
                    break;
                }
                case BIDataType::S32:
                {
                    /* Up-conversion QASYMM8_SIGNED -> S32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const int8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int32_t *>(dst.ptr());
                            int        x       = window_start_x;

                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int8x16_t texels_s8 = vld1q_s8(src_ptr + x);

                                const int16x8x2_t texels = {
                                    {vmovl_s8(vget_low_s8(texels_s8)), vmovl_s8(vget_high_s8(texels_s8))}};

                                vst1q_s32(dst_ptr + x, vmovl_s16(vget_low_s16(texels.val[0])));
                                vst1q_s32(dst_ptr + x + 4, vmovl_s16(vget_high_s16(texels.val[0])));
                                vst1q_s32(dst_ptr + x + 8, vmovl_s16(vget_low_s16(texels.val[1])));
                                vst1q_s32(dst_ptr + x + 12, vmovl_s16(vget_high_s16(texels.val[1])));
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
                case BIDataType::F32:
                {
                    /* Up-conversion QASYMM8_SIGNED -> F32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const int8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<float *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int8x16_t texels_s8 = vld1q_s8(src_ptr + x);

                                const int16x8x2_t texels = {
                                    {vmovl_s8(vget_low_s8(texels_s8)), vmovl_s8(vget_high_s8(texels_s8))}};
                                vst1q_f32(dst_ptr + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[0]))));
                                vst1q_f32(dst_ptr + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[0]))));
                                vst1q_f32(dst_ptr + x + 8, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[1]))));
                                vst1q_f32(dst_ptr + x + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[1]))));
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
                case BIDataType::F16:
                {
                    /* Up-conversion QASYMM8_SIGNED -> F16 */
                    BI_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
                    uk->ukernel(_src, _dst, info, _policy, window);
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        }

        case BIDataType::QASYMM8:
        case BIDataType::U8:
        {
            switch (_dst->info()->data_type())
            {
                case BIDataType::S16:
                {
                    /* Up-conversion U8 -> S16 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const uint8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int16_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint8x16_t texels_u8 = vld1q_u8(src_ptr + x);

                                const int16x8x2_t texels = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))),
                                                             vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8)))}};

                                vst1q_s16(dst_ptr + x, texels.val[0]);
                                vst1q_s16(dst_ptr + x + 8, texels.val[1]);
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
                case BIDataType::S32:
                {
                    /* Up-conversion U8 -> S32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const uint8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int32_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint8x16_t texels_u8 = vld1q_u8(src_ptr + x);

                                const int16x8x2_t texels = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))),
                                                             vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8)))}};

                                vst1q_s32(dst_ptr + x, vmovl_s16(vget_low_s16(texels.val[0])));
                                vst1q_s32(dst_ptr + x + 4, vmovl_s16(vget_high_s16(texels.val[0])));
                                vst1q_s32(dst_ptr + x + 8, vmovl_s16(vget_low_s16(texels.val[1])));
                                vst1q_s32(dst_ptr + x + 12, vmovl_s16(vget_high_s16(texels.val[1])));
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                *(dst_ptr + x) = static_cast<uint32_t>(*(src_ptr + x));
                            }
                        },
                        src, dst);
                    break;
                }
                case BIDataType::F32:
                {
                    /* Up-conversion U8 -> F32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const uint8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<float *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint8x16_t texels_u8 = vld1q_u8(src_ptr + x);

                                const int16x8x2_t texels = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))),
                                                             vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8)))}};
                                vst1q_f32(dst_ptr + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[0]))));
                                vst1q_f32(dst_ptr + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[0]))));
                                vst1q_f32(dst_ptr + x + 8, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[1]))));
                                vst1q_f32(dst_ptr + x + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[1]))));
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                *(dst_ptr + x) = static_cast<uint32_t>(*(src_ptr + x));
                            }
                        },
                        src, dst);
                    break;
                }
                case BIDataType::F16:
                {
                    /* Up-conversion U8 -> FP16 */
                    BI_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
                    uk->ukernel(_src, _dst, info, _policy, window);
                    break;
                }
                case BIDataType::U16:
                {
                    /* Up-conversion U8 -> U16 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const uint8_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<uint16_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint8x16_t texels_u8 = vld1q_u8(src_ptr + x);

                                const uint16x8x2_t texels = {
                                    {vmovl_u8(vget_low_u8(texels_u8)), vmovl_u8(vget_high_u8(texels_u8))}};

                                vst1q_u16(dst_ptr + x, texels.val[0]);
                                vst1q_u16(dst_ptr + x + 8, texels.val[1]);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                *(dst_ptr + x) = static_cast<uint16_t>(*(src_ptr + x));
                            }
                        },
                        src, dst);
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        }
        case BIDataType::S16:
        {
            switch (_dst->info()->data_type())
            {
                case BIDataType::QASYMM8_SIGNED:
                {
                    /* Down-conversion S16 -> QASYMM8_SIGNED */
                    if (BIConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int16_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int16x8x2_t texels = {{vld1q_s16(src_ptr + x), vld1q_s16(src_ptr + x + 8)}};

                                    vst1q_s8(dst_ptr + x,
                                             vcombine_s8(vqmovn_s16(texels.val[0]), vqmovn_s16(texels.val[1])));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = utils::cast::saturate_cast<int8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    else
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int16_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int16x8x2_t texels = {{vld1q_s16(src_ptr + x), vld1q_s16(src_ptr + x + 8)}};

                                    vst1q_s8(dst_ptr + x,
                                             vcombine_s8(vmovn_s16(texels.val[0]), vmovn_s16(texels.val[1])));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = static_cast<int8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    break;
                }
                case BIDataType::U8:
                {
                    /* Down-conversion S16 -> U8 */
                    if (BIConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int16_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int16x8x2_t texels = {{vld1q_s16(src_ptr + x), vld1q_s16(src_ptr + x + 8)}};

                                    vst1q_u8(dst_ptr + x,
                                             vcombine_u8(vqmovun_s16(texels.val[0]), vqmovun_s16(texels.val[1])));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    else
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int16_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int16x8x2_t texels = {{vld1q_s16(src_ptr + x), vld1q_s16(src_ptr + x + 8)}};

                                    vst1q_u8(dst_ptr + x, vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(texels.val[0])),
                                                                      vmovn_u16(vreinterpretq_u16_s16(texels.val[1]))));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = static_cast<uint8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    break;
                }
                case BIDataType::S32:
                {
                    /* Up-conversion S16 -> S32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const int16_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int32_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int16x8x2_t texels = {{vld1q_s16(src_ptr + x), vld1q_s16(src_ptr + x + 8)}};

                                const int32x4x4_t texels_s32 = {
                                    {vmovl_s16(vget_low_s16(texels.val[0])), vmovl_s16(vget_high_s16(texels.val[0])),
                                     vmovl_s16(vget_low_s16(texels.val[1])), vmovl_s16(vget_high_s16(texels.val[1]))}};

                                vst1q_s32(dst_ptr + x, texels_s32.val[0]);
                                vst1q_s32(dst_ptr + x + 4, texels_s32.val[1]);
                                vst1q_s32(dst_ptr + x + 8, texels_s32.val[2]);
                                vst1q_s32(dst_ptr + x + 12, texels_s32.val[3]);
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
            break;
        }

        case BIDataType::U16:
        {
            switch (_dst->info()->data_type())
            {
                case BIDataType::U8:
                {
                    /* Down-conversion U16 -> U8 */
                    if (BIConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const uint16_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const uint16x8x2_t texels = {{vld1q_u16(src_ptr + x), vld1q_u16(src_ptr + x + 8)}};

                                    vst1q_u8(dst_ptr + x,
                                             vcombine_u8(vqmovn_u16(texels.val[0]), vqmovn_u16(texels.val[1])));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    else
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const uint16_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const uint16x8x2_t texels = {{vld1q_u16(src_ptr + x), vld1q_u16(src_ptr + x + 8)}};

                                    vst1q_u8(dst_ptr + x,
                                             vcombine_u8(vmovn_u16(texels.val[0]), vmovn_u16(texels.val[1])));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = static_cast<uint8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    break;
                }
                case BIDataType::U32:
                {
                    /* Up-conversion U16 -> U32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const uint16_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<uint32_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint16x8x2_t texels = {{vld1q_u16(src_ptr + x), vld1q_u16(src_ptr + x + 8)}};

                                vst1q_u32(dst_ptr + x, vmovl_u16(vget_low_u16(texels.val[0])));
                                vst1q_u32(dst_ptr + x + 4, vmovl_u16(vget_high_u16(texels.val[0])));
                                vst1q_u32(dst_ptr + x + 8, vmovl_u16(vget_low_u16(texels.val[1])));
                                vst1q_u32(dst_ptr + x + 12, vmovl_u16(vget_high_u16(texels.val[1])));
                            }
                            // Compute left-over elements
                            for (; x < window_end_x; ++x)
                            {
                                *(dst_ptr + x) = static_cast<uint32_t>(*(src_ptr + x));
                            }
                        },
                        src, dst);
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        }
        case BIDataType::F16:
        {
            /* conversion F16 -> any data type */
            BI_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
            uk->ukernel(_src, _dst, info, _policy, window);
            break;
        }
        case BIDataType::F32:
            switch (_dst->info()->data_type())
            {
                case BIDataType::F16:
                {
                    /* Down-conversion F32 -> F16 */
                    BI_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
                    uk->ukernel(_src, _dst, info, _policy, window);
                    break;
                }
                case BIDataType::S32:
                {
                    /* Conversion F32 -> S32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const float *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int32_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const float32x4x4_t texels = {{
                                    vld1q_f32(src_ptr + x),
                                    vld1q_f32(src_ptr + x + 4),
                                    vld1q_f32(src_ptr + x + 8),
                                    vld1q_f32(src_ptr + x + 12),
                                }};

                                vst1q_s32(dst_ptr + x, vcvtq_s32_f32(texels.val[0]));
                                vst1q_s32(dst_ptr + x + 4, vcvtq_s32_f32(texels.val[1]));
                                vst1q_s32(dst_ptr + x + 8, vcvtq_s32_f32(texels.val[2]));
                                vst1q_s32(dst_ptr + x + 12, vcvtq_s32_f32(texels.val[3]));
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
                case BIDataType::QASYMM8:
                case BIDataType::U8:
                {
                    /* Down-conversion F32 -> QASYMM8, U8 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const float *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const float32x4x4_t texels = {{
                                    vld1q_f32(src_ptr + x),
                                    vld1q_f32(src_ptr + x + 4),
                                    vld1q_f32(src_ptr + x + 8),
                                    vld1q_f32(src_ptr + x + 12),
                                }};

                                vst1_u8(dst_ptr + x,
                                        vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(texels.val[0])),
                                                                vqmovun_s32(vcvtq_s32_f32(texels.val[1])))));
                                vst1_u8(dst_ptr + x + 8,
                                        vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(texels.val[2])),
                                                                vqmovun_s32(vcvtq_s32_f32(texels.val[3])))));
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
                case BIDataType::QASYMM8_SIGNED:
                {
                    /* Down-conversion F32 -> QASYMM8_SIGNED */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const float *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const float32x4x4_t texels = {{
                                    vld1q_f32(src_ptr + x),
                                    vld1q_f32(src_ptr + x + 4),
                                    vld1q_f32(src_ptr + x + 8),
                                    vld1q_f32(src_ptr + x + 12),
                                }};

                                vst1_s8(dst_ptr + x,
                                        vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(texels.val[0])),
                                                                vqmovn_s32(vcvtq_s32_f32(texels.val[1])))));
                                vst1_s8(dst_ptr + x + 8,
                                        vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(texels.val[2])),
                                                                vqmovn_s32(vcvtq_s32_f32(texels.val[3])))));
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

                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        case BIDataType::S32:
            switch (_dst->info()->data_type())
            {
#if __aarch64__
                case BIDataType::S64:
                {
                    convert64<int32_t, int64_t>(src, dst, win, window_start_x, window_end_x, window_step_x);
                    break;
                }
#endif // __aarch64__
                case BIDataType::F16:
                {
                    /* Down-conversion S32 -> F16 */
                    BI_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
                    uk->ukernel(_src, _dst, info, _policy, window);
                    break;
                }
                case BIDataType::F32:
                {
                    /* Conversion S32 -> F32 */
                    execute_window_loop(
                        win,
                        [&](const BICoordinates &)
                        {
                            const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
                            const auto dst_ptr = reinterpret_cast<float *>(dst.ptr());

                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int32x4x4_t texels = {{
                                    vld1q_s32(src_ptr + x),
                                    vld1q_s32(src_ptr + x + 4),
                                    vld1q_s32(src_ptr + x + 8),
                                    vld1q_s32(src_ptr + x + 12),
                                }};

                                vst1q_f32(dst_ptr + x, vcvtq_f32_s32(texels.val[0]));
                                vst1q_f32(dst_ptr + x + 4, vcvtq_f32_s32(texels.val[1]));
                                vst1q_f32(dst_ptr + x + 8, vcvtq_f32_s32(texels.val[2]));
                                vst1q_f32(dst_ptr + x + 12, vcvtq_f32_s32(texels.val[3]));
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
                case BIDataType::QASYMM8_SIGNED:
                {
                    /* Down-conversion S32 -> QASYMM8_SIGNED */
                    if (BIConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int32x4x4_t texels = {{
                                        vld1q_s32(src_ptr + x),
                                        vld1q_s32(src_ptr + x + 4),
                                        vld1q_s32(src_ptr + x + 8),
                                        vld1q_s32(src_ptr + x + 12),
                                    }};
                                    vst1_s8(dst_ptr + x, vqmovn_s16(vcombine_s16(vqmovn_s32(texels.val[0]),
                                                                                 vqmovn_s32(texels.val[1]))));
                                    vst1_s8(dst_ptr + x + 8, vqmovn_s16(vcombine_s16(vqmovn_s32(texels.val[2]),
                                                                                     vqmovn_s32(texels.val[3]))));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = utils::cast::saturate_cast<int8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    else
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int32x4x4_t texels = {{vld1q_s32(src_ptr + x), vld1q_s32(src_ptr + x + 4),
                                                                 vld1q_s32(src_ptr + x + 8),
                                                                 vld1q_s32(src_ptr + x + 12)}};

                                    vst1_s8(dst_ptr + x, vmovn_s16(vcombine_s16(vmovn_s32(texels.val[0]),
                                                                                vmovn_s32(texels.val[1]))));
                                    vst1_s8(dst_ptr + x + 8, vmovn_s16(vcombine_s16(vmovn_s32(texels.val[2]),
                                                                                    vmovn_s32(texels.val[3]))));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = static_cast<int8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    break;
                }
                case BIDataType::QASYMM8:
                case BIDataType::U8:
                {
                    /* Down-conversion S32 -> U8 */
                    if (BIConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int32x4x4_t texels = {{vld1q_s32(src_ptr + x), vld1q_s32(src_ptr + x + 4),
                                                                 vld1q_s32(src_ptr + x + 8),
                                                                 vld1q_s32(src_ptr + x + 12)}};
                                    vst1_u8(dst_ptr + x, vqmovn_u16(vcombine_u16(vqmovun_s32(texels.val[0]),
                                                                                 vqmovun_s32(texels.val[1]))));
                                    vst1_u8(dst_ptr + x + 8, vqmovn_u16(vcombine_u16(vqmovun_s32(texels.val[2]),
                                                                                     vqmovun_s32(texels.val[3]))));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    else
                    {
                        execute_window_loop(
                            win,
                            [&](const BICoordinates &)
                            {
                                const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
                                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                {
                                    const int32x4x4_t texels = {{vld1q_s32(src_ptr + x), vld1q_s32(src_ptr + x + 4),
                                                                 vld1q_s32(src_ptr + x + 8),
                                                                 vld1q_s32(src_ptr + x + 12)}};

                                    vst1_u8(dst_ptr + x,
                                            vmovn_u16(vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(texels.val[0])),
                                                                   vmovn_u32(vreinterpretq_u32_s32(texels.val[1])))));
                                    vst1_u8(dst_ptr + x + 8,
                                            vmovn_u16(vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(texels.val[2])),
                                                                   vmovn_u32(vreinterpretq_u32_s32(texels.val[3])))));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x)
                                {
                                    *(dst_ptr + x) = static_cast<uint8_t>(*(src_ptr + x));
                                }
                            },
                            src, dst);
                    }
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("dst data type not supported");
            }
            break;
        default:
            BI_COMPUTE_ERROR("Not supported");
    }
}

const char *BICpuCastKernel::name() const
{
    return "CpuCastKernel.cpp";
}

const std::vector<BICpuCastKernel::CastKernel> &BICpuCastKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels

} // namespace cpu

} // namespace BatmanInfer
