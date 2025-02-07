//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <data/core/bi_helpers.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/bi_ne_asymm.hpp>
#include <data/core/neon/wrapper/intrinsics/intrinsics.h>

namespace BatmanInfer {
    namespace cpu {
        constexpr auto window_step = 16;

        template<typename T>
        inline float32x4x4_t load_value(const T *input_ptr) {
            using Tx16_t = typename wrapper::traits::neon_vector<T, 16>::type;
            return convert_to_float32x4x4<Tx16_t>(wrapper::vloadq(input_ptr));
        }

        template<>
        inline float32x4x4_t load_value(const float *input_ptr) {
            return {wrapper::vloadq(input_ptr), wrapper::vloadq(input_ptr + 4), wrapper::vloadq(input_ptr + 8),
                    wrapper::vloadq(input_ptr + 12)};
        }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        template<>
        inline float32x4x4_t load_value(const float16_t *input_ptr) {
            return {vcvt_f32_f16(wrapper::vload(input_ptr)), vcvt_f32_f16(wrapper::vload(input_ptr + 4)),
                    vcvt_f32_f16(wrapper::vload(input_ptr + 8)), vcvt_f32_f16(wrapper::vload(input_ptr + 12))};
        }

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        template<typename element_type>
        using vector_type = wrapper::traits::neon_vector_t<element_type, window_step>;

        template<typename quantized_type>
        inline vector_type<quantized_type>
        vquantize_qasymm8(const float32x4x4_t &qv, const BIUniformQuantizationInfo &qi);

        template<>
        inline vector_type<uint8_t>
        vquantize_qasymm8<uint8_t>(const float32x4x4_t &qv, const BIUniformQuantizationInfo &qi) {
            return vquantize(qv, qi);
        }

        template<>
        inline vector_type<int8_t>
        vquantize_qasymm8<int8_t>(const float32x4x4_t &qv, const BIUniformQuantizationInfo &qi) {
            return vquantize_signed(qv, qi);
        }

        template<typename quantized_type>
        inline vector_type<quantized_type>
        vquantize_qasymm8(const float32x4x4_t &qv, const BIUniformRequantizationInfo &qi);

        template<>
        inline vector_type<uint8_t>
        vquantize_qasymm8<uint8_t>(const float32x4x4_t &qv, const BIUniformRequantizationInfo &qi) {
            return vquantize(qv, qi);
        }

        template<>
        inline vector_type<int8_t>
        vquantize_qasymm8<int8_t>(const float32x4x4_t &qv, const BIUniformRequantizationInfo &qi) {
            return vquantize_signed(qv, qi);
        }

        template<typename TOut, typename = typename std::enable_if<std::is_signed<TOut>::value, bool>::type>
        inline int8x16_t recombine_8_16(int16x8_t lower, int16x8_t upper) {
            return wrapper::vcombine(wrapper::vqmovn(lower), wrapper::vqmovn(upper));
        }

        template<typename TOut, typename = typename std::enable_if<std::is_unsigned<TOut>::value, bool>::type>
        inline uint8x16_t recombine_8_16(int16x8_t lower, int16x8_t upper) {
            return wrapper::vcombine(wrapper::vqmovun(lower), wrapper::vqmovun(upper));
        }

        template<typename TIn, typename TOut>
        void run_quantize_qsymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            const BIUniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
            BIUniformQuantizationInfo uqinfo = dst->info()->quantization_info().uniform();
            uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);
            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        auto input_ptr = reinterpret_cast<const TIn *>(input.ptr());
                        auto output_ptr = reinterpret_cast<TOut *>(output.ptr());
                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step); x += window_step) {
                            wrapper::vstore(&output_ptr[x], vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), uqinfo));
                        }
                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            output_ptr[x] = quantize_qsymm8(input_ptr[x], dst->info()->quantization_info());
                        }
                    },
                    input, output);
        }

        template<typename TIn, typename TOut>
        void run_requantize_offset_only_convert(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            // Calculate output offset difference.
            const BIUniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
            BIUniformQuantizationInfo uqinfo = dst->info()->quantization_info().uniform();
            uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);

            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Duplicate offset in signed vector format
            const int8x16_t offset = wrapper::vdup_n(static_cast<int8_t>(uqinfo.offset),
                                                     wrapper::traits::vector_128_tag{});

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);
            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        auto input_ptr = reinterpret_cast<const TIn *>(input.ptr());
                        auto output_ptr = reinterpret_cast<TOut *>(output.ptr());
                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step); x += window_step) {
                            const wrapper::traits::neon_vector_t<TIn, window_step> qv =
                                    wrapper::vloadq(input_ptr + x); // load 128 bit vector of 8 bit datatype

                            // Signed addition.
                            auto res = vaddq_s8(reinterpret_cast<int8x16_t>(qv), offset);

                            // Output is dependent on datatype.
                            wrapper::vstore(&output_ptr[x],
                                            reinterpret_cast<wrapper::traits::neon_vector_t<TOut, window_step>>(res));
                        }
                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            auto result = uqinfo.offset + static_cast<int32_t>(input_ptr[x]);
                            output_ptr[x] = static_cast<TOut>(result);
                        }
                    },
                    input, output);
        }

        template<typename TIn, typename TOut>
        void run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            const BIUniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
            BIUniformQuantizationInfo uqinfo = dst->info()->quantization_info().uniform();
            uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Duplicate offset in signed vector format
            const int16x8_t offset = wrapper::vdup_n(static_cast<int16_t>(uqinfo.offset),
                                                     wrapper::traits::vector_128_tag{});

            const int32_t low_bound = (dst->info()->data_type() == BIDataType::QASYMM8) ? 0 : -128;
            const int32_t upper_bound = (dst->info()->data_type() == BIDataType::QASYMM8) ? 255 : 127;

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);
            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        auto input_ptr = reinterpret_cast<const TIn *>(input.ptr());
                        TOut *output_ptr = reinterpret_cast<TOut *>(output.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step); x += window_step) {
                            const auto qv = wrapper::vloadq(input_ptr + x); // load 128 bit vector of 8 bit datatype
                            int16x8_t lower = reinterpret_cast<int16x8_t>(wrapper::vmovl(wrapper::vgetlow(qv)));
                            int16x8_t upper = reinterpret_cast<int16x8_t>(wrapper::vmovl(wrapper::vgethigh(qv)));

                            // Signed addition.
                            lower = wrapper::vqadd(lower, offset);
                            upper = wrapper::vqadd(upper, offset);

                            // Output is dependent on datatype.
                            auto res = recombine_8_16<TOut>(lower, upper);
                            wrapper::vstore(&output_ptr[x], res);
                        }
                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            // Add offset and clamp result to within the range of the output datatype.
                            int32_t result = uqinfo.offset + static_cast<int32_t>(input_ptr[x]);
                            result = misc::utility::clamp<int32_t>(result, low_bound, upper_bound);

                            // Cast result to output datatype.
                            output_ptr[x] = static_cast<TOut>(result);
                        }
                    },
                    input, output);
        }

        template<typename TIn, typename TOut>
        void run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            constexpr bool is_8bit_int = std::is_same<TIn, int8_t>::value || std::is_same<TIn, uint8_t>::value;

            const BIUniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
            BIUniformQuantizationInfo uqinfo = dst->info()->quantization_info().uniform();
            BIUniformRequantizationInfo reqinfo(1.f, 0);

            if (is_8bit_int) {
                reqinfo = compute_requantization_scale_float_offset(uqinfo_in, uqinfo);
            }

#ifdef __aarch64__
            constexpr BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
            constexpr BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_ZERO;
#endif //__aarch64__

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);
            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        auto input_ptr = reinterpret_cast<const TIn *>(input.ptr());
                        auto output_ptr = reinterpret_cast<TOut *>(output.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step); x += window_step) {
                            if (is_8bit_int) {
                                wrapper::vstore(&output_ptr[x],
                                                vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), reqinfo));
                            } else {
                                wrapper::vstore(&output_ptr[x],
                                                vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), uqinfo));
                            }
                        }
                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            if (is_8bit_int) {
                                output_ptr[x] = BIQasymm8QuantizationHelper<TOut>::quantize(input_ptr[x], reqinfo,
                                                                                            rounding_policy);
                            } else {
                                output_ptr[x] = BIQasymm8QuantizationHelper<TOut>::quantize(input_ptr[x], uqinfo,
                                                                                            rounding_policy);
                            }
                        }
                    },
                    input, output);
        }

        template<typename T>
        void run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            const BIUniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
            BIUniformQuantizationInfo uqinfo = dst->info()->quantization_info().uniform();
            if (is_data_type_quantized_asymmetric(src->info()->data_type())) {
                uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);
            }
#ifdef __aarch64__
            constexpr BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
            constexpr BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_ZERO;
#endif //__aarch64__

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);
            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        auto input_ptr = reinterpret_cast<const T *>(input.ptr());
                        auto output_ptr = reinterpret_cast<uint16_t *>(output.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step); x += window_step) {
                            uint16x8x2_t tmp = vquantize_qasymm16(load_value(&input_ptr[x]), uqinfo);
                            vst1q_u16(&output_ptr[x], tmp.val[0]);
                            vst1q_u16(&output_ptr[x + 8], tmp.val[1]);
                        }
                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            output_ptr[x] = quantize_qasymm16(input_ptr[x], uqinfo, rounding_policy);
                        }
                    },
                    input, output);
        }
    } // namespace cpu
} // namespace BatmanInfer