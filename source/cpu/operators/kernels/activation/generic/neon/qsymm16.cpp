//
// Created by Mason on 2025/1/10.
//

#include <cpu/kernels/activation/generic/neon/qsymm16.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_qsymm16_activation(const BIITensor *src,
                                     BIITensor *dst,
                                     const BIActivationLayerInfo &act_info,
                                     const BIWindow &window) {
            constexpr int window_step_x = 8;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());
            const BIActivationLayerInfo::ActivationFunction act = act_info.activation();

            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);

            const BIUniformQuantizationInfo qi_in = src->info()->quantization_info().uniform();
            const BIUniformQuantizationInfo qi_out = dst->info()->quantization_info().uniform();
            const auto vconst_1 = vdupq_n_f32(1.f);
            const float32x4_t va_f32 = vdupq_n_f32(act_info.a());
            const float32x4_t vb_f32 = vdupq_n_f32(act_info.b());
            const float a_f32 = act_info.a();
            const float b_f32 = act_info.b();

            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        const auto input_ptr = reinterpret_cast<const qsymm16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<qsymm16_t *>(output.ptr());

                        wrapper::traits::neon_bitvector_t<qsymm16_t, wrapper::traits::BitWidth::W128> tmp;
                        BI_COMPUTE_UNUSED(tmp);

                        // Compute S elements per iteration
                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const auto vin = wrapper::vloadq(input_ptr + x);
                            if (act == BIActivationLayerInfo::ActivationFunction::LOGISTIC) {
                                // De-quantize
                                const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                                // Perform activation
                                const float32x4x2_t tmp_dep = {{
                                                                       wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1,
                                                                                                             wrapper::vexpq(
                                                                                                                     wrapper::vneg(
                                                                                                                             vin_deq.val[0])))),
                                                                       wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1,
                                                                                                             wrapper::vexpq(
                                                                                                                     wrapper::vneg(
                                                                                                                             vin_deq.val[1])))),
                                                               }};
                                // Re-quantize to new output space
                                tmp = vquantize_int16(tmp_dep, qi_out.scale);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::TANH) {
                                // De-quantize
                                const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                                // Perform activation
                                const float32x4x2_t tmp_dep = {{
                                                                       wrapper::vmul(va_f32, wrapper::vtanh(
                                                                               wrapper::vmul(vin_deq.val[0], vb_f32))),
                                                                       wrapper::vmul(va_f32, wrapper::vtanh(
                                                                               wrapper::vmul(vin_deq.val[1], vb_f32))),
                                                               }};
                                // Re-quantize to new output space
                                tmp = vquantize_int16(tmp_dep, qi_out.scale);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU) {
                                // De-quantize
                                const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                                // Perform activation
                                const float32x4x2_t tmp_dep = {
                                        {wrapper::vmin(va_f32, wrapper::vmax(vb_f32, vin_deq.val[0])),
                                         wrapper::vmin(va_f32, wrapper::vmax(vb_f32, vin_deq.val[1]))}};
                                // Re-quantize to new output space
                                tmp = vquantize_int16(tmp_dep, qi_out.scale);
                            } else {
                                BI_COMPUTE_ERROR("Unsupported activation function");
                            }
                            wrapper::vstore(output_ptr + x, tmp);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            qsymm16_t in = *(reinterpret_cast<const qsymm16_t *>(input_ptr + x));
                            qsymm16_t tmp = 0;
                            if (act == BIActivationLayerInfo::ActivationFunction::LOGISTIC) {
                                float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                                tmp_f = 1.f / (1.f + std::exp(-tmp_f));
                                tmp = quantize_qsymm16(tmp_f, qi_out);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::TANH) {
                                float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                                tmp_f = a_f32 * std::tanh(b_f32 * tmp_f);
                                tmp = quantize_qsymm16(tmp_f, qi_out);
                            } else if (act == BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU) {
                                float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                                tmp_f = std::min<float>(a_f32, std::max<float>(b_f32, tmp_f));
                                tmp = quantize_qsymm16(tmp_f, qi_out);
                            } else {
                                BI_COMPUTE_ERROR("Unsupported activation function");
                            }
                            *(output_ptr + x) = tmp;
                        }
                    },
                    input, output);
        }
    } // namespace cpu
}