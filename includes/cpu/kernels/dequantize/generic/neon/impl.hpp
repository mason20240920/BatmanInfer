//
// Created by Mason on 2025/2/8.
//

#pragma once

#include "data/core/bi_helpers.hpp"
#include <data/core/bi_window.hpp>
#include <data/core/quantization_info.hpp>

#include <data/core/neon/bi_ne_asymm.hpp>
#include <data/core/neon/bi_ne_symm.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
        template<typename T>
        inline void store_result(T *ptr, const float32x4x4_t &v) {
            BI_COMPUTE_UNUSED(ptr, v);
        }

        template<>
        inline void store_result<float>(float *ptr, const float32x4x4_t &v) {
            wrapper::vstore(ptr, v.val[0]);
            wrapper::vstore(ptr + 4, v.val[1]);
            wrapper::vstore(ptr + 8, v.val[2]);
            wrapper::vstore(ptr + 12, v.val[3]);
        }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        /**
         * 针对 float16_t 类型的特化版本，用于将 float32x4x4_t 类型的数据存储到内存中。
         * @param ptr 指向存储位置的指针。
         * @param v 包含 4 个 float32x4 向量的数据结构。
         * 使用 NEON 指令将 float32 数据转换为 float16 并存储。
         */
        template<>
        inline void store_result<float16_t>(float16_t *ptr, const float32x4x4_t &v) {
            // 将前两个 float32x4 向量转换为 float16，并合并为一个 float16x8 向量后存储。
            wrapper::vstore(ptr, vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1])));
            // 将后两个 float32x4 向量转换为 float16，并合并为一个 float16x8 向量后存储。
            wrapper::vstore(ptr + 8, vcombine_f16(vcvt_f16_f32(v.val[2]), vcvt_f16_f32(v.val[3])));
        }

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

        template<typename T>
        inline void store_result(T *ptr, const float32x4x2_t &v) {
            BI_COMPUTE_UNUSED(ptr, v);
        }

        /**
         * 针对 float 类型的特化版本，用于将 float32x4x2_t 类型的数据存储到内存中。
         * @param ptr 指向存储位置的指针。
         * @param v 包含 2 个 float32x4 向量的数据结构。
         */
        template<>
        inline void store_result<float>(float *ptr, const float32x4x2_t &v) {
            // 将第一个 float32x4 向量存储到内存中。
            wrapper::vstore(ptr, v.val[0]);
            // 将第二个 float32x4 向量存储到内存中，偏移量为 4。
            wrapper::vstore(ptr + 4, v.val[1]);
        }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        /**
         * 针对 float16_t 类型的特化版本，用于将 float32x4x2_t 类型的数据存储到内存中。
         * @param ptr 指向存储位置的指针
         * @param v 包含 2 个 float32x4 向量的数据结构
         */
        template<>
        inline void store_result<float16_t>(float16_t *ptr, const float32x4x2_t &v) {
            wrapper::vstore(ptr, vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1])));
        }

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

        template<typename TOut, typename TIn>
        void run_dequantization_qasymm8(const BIITensor *input,
                                        BIITensor *output,
                                        const BIWindow &window) {
            const BIUniformQuantizationInfo &qinfo = input->info()->quantization_info().uniform();
            const float scale = qinfo.scale;
            const int32_t offset = qinfo.offset;

            const int window_step_x = 16;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Create iterators
            BIIterator in(input, win_collapsed);
            BIIterator out(output, win_collapsed);

            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        const auto in_ptr = reinterpret_cast<const TIn *>(in.ptr());
                        const auto out_ptr = reinterpret_cast<TOut *>(out.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const auto vin = wrapper::vloadq(in_ptr + x);
                            const auto vdeq = vdequantize(vin, scale, offset);

                            store_result(reinterpret_cast<TOut *>(out_ptr + x), vdeq);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            auto val = *(in_ptr + x);
                            *(out_ptr + x) = static_cast<TOut>(BIQasymm8QuantizationHelper<TIn>::dequantize(val,
                                                                                                            qinfo));
                        }
                    },
                    in, out);
        }

        template<typename T>
        void
        run_dequantization_qsymm8_per_channel_nchw(const BIITensor *input, BIITensor *output, const BIWindow &window) {
            const auto scale = input->info()->quantization_info().scale();

            const int window_step_x = 16;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            // Reset first dimension to handle tail calculations manually
            BIWindow win(window);
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Create iterators
            BIIterator in(input, win);
            BIIterator out(output, win);

            execute_window_loop(
                    win,
                    [&](const BICoordinates &id) {
                        const auto in_ptr = reinterpret_cast<const int8_t *>(in.ptr());
                        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const auto vin = wrapper::vloadq(in_ptr + x);
                            const auto vdeq = vdequantize(vin, scale[id.z()]);

                            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            int8_t val = *(in_ptr + x);
                            *(out_ptr + x) = static_cast<T>(dequantize(val, scale[id.z()]));
                        }
                    },
                    in, out);
        }

        template<typename T>
        void
        run_dequantization_qsymm8_per_channel_nhwc(const BIITensor *input, BIITensor *output, const BIWindow &window) {
            const auto scale = input->info()->quantization_info().scale();

            const int window_step_x = 16;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            // Reset first dimension to handle tail calculations manually
            BIWindow win(window);
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Create iterators
            BIIterator in(input, win);
            BIIterator out(output, win);

            execute_window_loop(
                    win,
                    [&](const BICoordinates &) {
                        const auto in_ptr = reinterpret_cast<const int8_t *>(in.ptr());
                        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const float32x4x4_t vscale = {
                                    {scale[x + 0], scale[x + 1], scale[x + 2], scale[x + 3], scale[x + 4],
                                     scale[x + 5], scale[x + 6], scale[x + 7], scale[x + 8], scale[x + 9],
                                     scale[x + 10], scale[x + 11], scale[x + 12], scale[x + 13],
                                     scale[x + 14], scale[x + 15]}};
                            const auto vin = wrapper::vloadq(in_ptr + x);
                            const auto vdeq = vdequantize(vin, vscale);

                            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            int8_t val = *(in_ptr + x);
                            *(out_ptr + x) = static_cast<T>(dequantize(val, scale[x]));
                        }
                    },
                    in, out);
        }

        template<typename T>
        void run_dequantization_qsymm8(const BIITensor *input, BIITensor *output, const BIWindow &window) {
            const BIUniformQuantizationInfo &qinfo = input->info()->quantization_info().uniform();
            const float scale = qinfo.scale;

            const int window_step_x = 16;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Create iterators
            BIIterator in(input, win_collapsed);
            BIIterator out(output, win_collapsed);

            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        const auto in_ptr = reinterpret_cast<const int8_t *>(in.ptr());
                        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const auto vin = wrapper::vloadq(in_ptr + x);
                            const auto vdeq = vdequantize(vin, scale);

                            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            int8_t val = *(in_ptr + x);
                            *(out_ptr + x) = static_cast<T>(dequantize(val, scale));
                        }
                    },
                    in, out);
        }

        template<typename T>
        void run_dequantization_qsymm16(const BIITensor *input, BIITensor *output, const BIWindow &window) {
            const BIUniformQuantizationInfo &qinfo = input->info()->quantization_info().uniform();
            const float scale = qinfo.scale;

            const int window_step_x = 8;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            // Collapse window and reset first dimension to handle tail calculations manually
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            // Create iterators
            BIIterator in(input, win_collapsed);
            BIIterator out(output, win_collapsed);

            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        const auto in_ptr = reinterpret_cast<const int16_t *>(in.ptr());
                        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

                        int x = window_start_x;
                        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                            const auto vin = wrapper::vloadq(in_ptr + x);
                            const auto vdeq = vdequantize_int16(vin, scale);

                            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
                        }

                        // Compute left-over elements
                        for (; x < window_end_x; ++x) {
                            int16_t val = *(in_ptr + x);
                            *(out_ptr + x) = static_cast<T>(dequantize_qsymm16(val, scale));
                        }
                    },
                    in, out);
        }

        template<typename T>
        void run_dequantization_core(const BIITensor *input, BIITensor *output, const BIWindow &window) {
            switch (input->info()->data_type()) {
                case BIDataType::QASYMM8:
                    run_dequantization_qasymm8<T, uint8_t>(input, output, window);
                    break;
                case BIDataType::QASYMM8_SIGNED:
                    run_dequantization_qasymm8<T, int8_t>(input, output, window);
                    break;
                case BIDataType::QSYMM8_PER_CHANNEL:
                    input->info()->data_layout() == BIDataLayout::NHWC
                    ? run_dequantization_qsymm8_per_channel_nhwc<T>(input, output, window)
                    : run_dequantization_qsymm8_per_channel_nchw<T>(input, output, window);
                    break;
                case BIDataType::QSYMM8:
                    run_dequantization_qsymm8<T>(input, output, window);
                    break;
                case BIDataType::QSYMM16:
                    run_dequantization_qsymm16<T>(input, output, window);
                    break;
                default:
                    BI_COMPUTE_ERROR("Unsupported data type.");
            }
        }
    } // namespace cpu
}