//
// Created by Mason on 2025/1/10.
//

#include "cpu/kernels/activation/generic/neon/qasymm8_signed.hpp"

namespace BatmanInfer {
    namespace cpu {
        void neon_qasymm8_signed_activation(const BIITensor *src,
                                            BIITensor *dst,
                                            const BIActivationLayerInfo &act_info,
                                            const BIWindow &window) {
            constexpr int window_step_x = 16;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());
            const BIActivationLayerInfo::ActivationFunction act = act_info.activation();

            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);

            const BIUniformQuantizationInfo qi_in = src->info()->quantization_info().uniform();
            const BIUniformQuantizationInfo qi_out = dst->info()->quantization_info().uniform();
            const qasymm8x16_signed_t va = vdupq_n_s8(quantize_qasymm8_signed(act_info.a(), qi_in));
            const qasymm8x16_signed_t vb = vdupq_n_s8(quantize_qasymm8_signed(act_info.b(), qi_in));
            const qasymm8_signed_t a = quantize_qasymm8_signed(act_info.a(), qi_in);
            const qasymm8_signed_t b = quantize_qasymm8_signed(act_info.b(), qi_in);
            const qasymm8_signed_t const_0 = quantize_qasymm8_signed(0.f, qi_in);
            const qasymm8x16_signed_t vconst_0 = vdupq_n_s8(const_0);
#ifndef __aarch64__
            const auto vconst_1     = vdupq_n_f32(1.f);
    const auto vconst_0_f32 = vdupq_n_f32(0.f);
#endif // __aarch64__
            const float32x4_t va_f32 = vdupq_n_f32(act_info.a());
            const float32x4_t vb_f32 = vdupq_n_f32(act_info.b());
            const float a_f32 = act_info.a();
            const float b_f32 = act_info.b();
            const auto const_6_f32 = vdupq_n_f32(6.f);
            const auto const_0_f32 = vdupq_n_f32(0.f);
            const auto const_3_f32 = vdupq_n_f32(3.f);
            const auto const_inv_6_f32 = vdupq_n_f32(0.166666667f);

            // Initialise scale/offset for re-quantization
            float s = qi_in.scale / qi_out.scale;
            float o = -qi_in.offset * s + qi_out.offset;
            float32x4_t vs = vdupq_n_f32(s);
            float32x4_t vo = vdupq_n_f32(o);

            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        const auto input_ptr = reinterpret_cast<const qasymm8_signed_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<qasymm8_signed_t *>(output.ptr());

                        wrapper::traits::neon_bitvector_t<qasymm8_signed_t, wrapper::traits::BitWidth::W128> tmp;

                        // Compute S elements per iteration
                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const auto vin = wrapper::vloadq(input_ptr + x);
                            if (act == BIActivationLayerInfo::ActivationFunction::RELU) {
                                // Perform activation
                                tmp = vmaxq_s8(vconst_0, vin);
                                // Re-quantize to new output space
                                tmp = vmlaq_qasymm8_signed<BIRoundingPolicy::TO_NEAREST_UP>(tmp, vs, vo);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU) {
                                // Perform activation
                                tmp = vminq_s8(va, vmaxq_s8(vconst_0, vin));
                                // Re-quantize to new output space
                                tmp = vmlaq_qasymm8_signed<BIRoundingPolicy::TO_NEAREST_UP>(tmp, vs, vo);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU) {
                                // Perform activation
                                tmp = vminq_s8(va, vmaxq_s8(vb, vin));
                                // Re-quantize to new output space
                                tmp = vmlaq_qasymm8_signed<BIRoundingPolicy::TO_NEAREST_UP>(tmp, vs, vo);
                            }
#ifndef __aarch64__ // LUT-based implementation is used for aarch64 instead.
                                else if (act == BIActivationLayerInfo::ActivationFunction::LOGISTIC)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize(vin, qi_in);
                    // Perform activation
                    const float32x4x4_t tmp_dep = {{
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[0])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[1])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[2])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[3])))),
                    }};
                    // Re-quantize to new output space
                    tmp = vquantize_signed(tmp_dep, qi_out);
                }
#endif // __aarch64__
                            else if (act == BIActivationLayerInfo::ActivationFunction::TANH) {
                                // De-quantize
                                const auto vin_deq = vdequantize(vin, qi_in);
                                // Perform activation
                                const float32x4x4_t tmp_dep = {{
                                                                       wrapper::vmul(va_f32, wrapper::vtanh(
                                                                               wrapper::vmul(vin_deq.val[0], vb_f32))),
                                                                       wrapper::vmul(va_f32, wrapper::vtanh(
                                                                               wrapper::vmul(vin_deq.val[1], vb_f32))),
                                                                       wrapper::vmul(va_f32, wrapper::vtanh(
                                                                               wrapper::vmul(vin_deq.val[2], vb_f32))),
                                                                       wrapper::vmul(va_f32, wrapper::vtanh(
                                                                               wrapper::vmul(vin_deq.val[3], vb_f32))),
                                                               }};
                                // Re-quantize to new output space
                                tmp = vquantize_signed(tmp_dep, qi_out);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::HARD_SWISH) {
                                // De-quantize
                                const auto vin_deq = vdequantize(vin, qi_in);
                                // Perform activation
                                const float32x4x4_t tmp_dep = {{
                                                                       wrapper::vmul(
                                                                               vin_deq.val[0],
                                                                               wrapper::vmul(
                                                                                       const_inv_6_f32,
                                                                                       wrapper::vmin(const_6_f32,
                                                                                                     wrapper::vmax(
                                                                                                             const_0_f32,
                                                                                                             wrapper::vadd(
                                                                                                                     vin_deq.val[0],
                                                                                                                     const_3_f32))))),
                                                                       wrapper::vmul(
                                                                               vin_deq.val[1],
                                                                               wrapper::vmul(
                                                                                       const_inv_6_f32,
                                                                                       wrapper::vmin(const_6_f32,
                                                                                                     wrapper::vmax(
                                                                                                             const_0_f32,
                                                                                                             wrapper::vadd(
                                                                                                                     vin_deq.val[1],
                                                                                                                     const_3_f32))))),
                                                                       wrapper::vmul(
                                                                               vin_deq.val[2],
                                                                               wrapper::vmul(
                                                                                       const_inv_6_f32,
                                                                                       wrapper::vmin(const_6_f32,
                                                                                                     wrapper::vmax(
                                                                                                             const_0_f32,
                                                                                                             wrapper::vadd(
                                                                                                                     vin_deq.val[2],
                                                                                                                     const_3_f32))))),
                                                                       wrapper::vmul(
                                                                               vin_deq.val[3],
                                                                               wrapper::vmul(
                                                                                       const_inv_6_f32,
                                                                                       wrapper::vmin(const_6_f32,
                                                                                                     wrapper::vmax(
                                                                                                             const_0_f32,
                                                                                                             wrapper::vadd(
                                                                                                                     vin_deq.val[3],
                                                                                                                     const_3_f32))))),
                                                               }};
                                // Re-quantize to new output space
                                tmp = vquantize_signed(tmp_dep, qi_out);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::LEAKY_RELU) {
                                const auto vin_deq = vdequantize(vin, qi_in);

#ifdef __aarch64__
                                const uint32x4x4_t pos_mask = {{
                                                                       wrapper::vcgtz(vin_deq.val[0]),
                                                                       wrapper::vcgtz(vin_deq.val[1]),
                                                                       wrapper::vcgtz(vin_deq.val[2]),
                                                                       wrapper::vcgtz(vin_deq.val[3]),
                                                               }};
#else  // __aarch64__
                                const uint32x4x4_t pos_mask = {{
                        wrapper::vcgt(vin_deq.val[0], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[1], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[2], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[3], vconst_0_f32),
                    }};
#endif // __aarch64__

                                const float32x4x4_t tmp_dep = {{
                                                                       wrapper::vbsl(pos_mask.val[0], vin_deq.val[0],
                                                                                     wrapper::vmul(va_f32,
                                                                                                   vin_deq.val[0])),
                                                                       wrapper::vbsl(pos_mask.val[1], vin_deq.val[1],
                                                                                     wrapper::vmul(va_f32,
                                                                                                   vin_deq.val[1])),
                                                                       wrapper::vbsl(pos_mask.val[2], vin_deq.val[2],
                                                                                     wrapper::vmul(va_f32,
                                                                                                   vin_deq.val[2])),
                                                                       wrapper::vbsl(pos_mask.val[3], vin_deq.val[3],
                                                                                     wrapper::vmul(va_f32,
                                                                                                   vin_deq.val[3])),
                                                               }};

                                tmp = vquantize_signed(tmp_dep, qi_out);
                            } else {
                                BI_COMPUTE_ERROR("Unsupported activation function");
                            }
                            wrapper::vstore(output_ptr + x, tmp);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            qasymm8_signed_t in = *(reinterpret_cast<const qasymm8_signed_t *>(input_ptr + x));
                            qasymm8_signed_t tmp = 0;
                            if (act == BIActivationLayerInfo::ActivationFunction::RELU) {
                                tmp = std::max(const_0, in);
                                tmp = misc::utility::clamp<int32_t, qasymm8_signed_t>(
                                        support::cpp11::lround(tmp * s + o));
                            } else if (act == BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU) {
                                tmp = std::min(a, std::max(const_0, in));
                                tmp = misc::utility::clamp<int32_t, qasymm8_signed_t>(
                                        support::cpp11::lround(tmp * s + o));
                            } else if (act == BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU) {
                                tmp = std::min(a, std::max(b, in));
                                tmp = misc::utility::clamp<int32_t, qasymm8_signed_t>(
                                        support::cpp11::lround(tmp * s + o));
                            }
#ifndef __aarch64__ // LUT-based implementation is used for aarch64 instead.
                                else if (act == BIActivationLayerInfo::ActivationFunction::LOGISTIC)
                {
                    float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                    tmp_f       = 1.f / (1.f + std::exp(-tmp_f));
                    tmp         = quantize_qasymm8_signed(tmp_f, qi_out);
                }
#endif // __aarch64__
                            else if (act == BIActivationLayerInfo::ActivationFunction::TANH) {
                                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                                tmp_f = a_f32 * std::tanh(b_f32 * tmp_f);
                                tmp = quantize_qasymm8_signed(tmp_f, qi_out);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::HARD_SWISH) {
                                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                                tmp_f = tmp_f * ((std::min(std::max((tmp_f + 3), 0.0f), 6.0f)) * 0.166666667f);
                                tmp = quantize_qasymm8_signed(tmp_f, qi_out);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::LEAKY_RELU) {
                                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                                tmp_f = tmp_f > 0 ? tmp_f : tmp_f * a_f32;
                                tmp = quantize_qasymm8_signed(tmp_f, qi_out);
                            } else {
                                BI_COMPUTE_ERROR("Unsupported activation function");
                            }
                            *(output_ptr + x) = tmp;
                        }
                    },
                    input, output);
        }
    }
}