//
// Created by Mason on 2025/4/3.
//

#pragma once

#include <data/core/bi_tensor_info.hpp>

#include <data/core/neon/bi_ne_asymm.hpp>
#include <cpu/kernels/select/generic/neon/impl.hpp>

namespace BatmanInfer::cpu {
    template<typename ScalarType, typename VectorType>
    void select_op(const BIITensor *cond,
                   const BIITensor *in1,
                   const BIITensor *in2,
                   BIITensor *out,
                   const BIWindow &window,
                   const int window_step_x,
                   const int window_start_x,
                   const int window_end_x,
                   const int limit,
                   VectorType (*condition_conversion)(const uint8_t *)) {
        BIWindow win = window;
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator condition(cond, win);
        BIIterator input1(in1, win);
        BIIterator input2(in2, win);
        BIIterator output(out, win);

        execute_window_loop(
            win,
            [&](const BICoordinates &) {
                auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());
                const auto condition_ptr = reinterpret_cast<const uint8_t *>(condition.ptr());
                const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
                const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());

                int x = window_start_x;
                for (; x <= limit; x += window_step_x) {
                    const auto c = (*condition_conversion)(condition_ptr + x);
                    const auto a = wrapper::vloadq(input1_ptr + x);
                    const auto b = wrapper::vloadq(input2_ptr + x);
                    wrapper::vstore(output_ptr + x, wrapper::vbsl(c, a, b));
                }
                for (; x < window_end_x; ++x) {
                    const auto c = *(condition_ptr + x);
                    const auto a = *(input1_ptr + x);
                    const auto b = *(input2_ptr + x);
                    *(output_ptr + x) = static_cast<bool>(c) ? a : b;
                }
            },
            condition, input1, input2, output);
    }

    template<typename ScalarType, typename VectorType>
    void select_op_8(const BIITensor *cond, const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                     const BIWindow &window) {
        const auto window_step_x = 16 / sizeof(ScalarType);
        const auto window_start_x = window.x().start();
        const auto window_end_x = window.x().end();

        select_op<ScalarType, VectorType>(
            cond, in1, in2, out, window, window_step_x, window_start_x, window_end_x, window_end_x - window_step_x,
            [](const uint8_t *condition_ptr) -> VectorType {
                static const auto zero =
                        wrapper::vdup_n(static_cast<uint8_t>(0), wrapper::traits::vector_128_tag());
                return wrapper::vcgt(wrapper::vloadq(condition_ptr), zero);
            });
    }

    template<typename ScalarType, typename VectorType>
    void select_op_16(const BIITensor *cond, const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                      const BIWindow &window) {
        const auto window_step_x = 16 / sizeof(ScalarType);
        const auto window_start_x = window.x().start();
        const auto window_end_x = window.x().end();

        select_op<ScalarType, VectorType>(
            cond, in1, in2, out, window, window_step_x, window_start_x, window_end_x, window_end_x - window_step_x,
            [](const uint8_t *condition_ptr) -> VectorType {
                static const auto zero =
                        wrapper::vdup_n(static_cast<uint16_t>(0), wrapper::traits::vector_128_tag());
                return wrapper::vcgt(wrapper::vmovl(wrapper::vload(condition_ptr)), zero);
            });
    }

    template<typename ScalarType, typename VectorType>
    void select_op_32(const BIITensor *cond, const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                      const BIWindow &window) {
        const auto window_step_x = 16 / sizeof(ScalarType);
        const auto window_start_x = window.x().start();
        const auto window_end_x = window.x().end();

        select_op<ScalarType, VectorType>(
            cond, in1, in2, out, window, window_step_x, window_start_x, window_end_x, window_end_x - window_step_x,
            [](const uint8_t *condition_ptr) -> VectorType {
                static const auto zero =
                        wrapper::vdup_n(static_cast<uint32_t>(0), wrapper::traits::vector_128_tag());
                return wrapper::vcgt(
                    wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vload(condition_ptr)))), zero);
            });
    }

    template<typename ScalarType>
    void select_op_not_same_rank(
        const BIITensor *cond, const BIITensor *in1, const BIITensor *in2, BIITensor *out, const BIWindow &window) {
        BI_COMPUTE_UNUSED(window);

        auto output_ptr = reinterpret_cast<ScalarType *>(out->buffer());
        const auto condition_ptr = reinterpret_cast<const uint8_t *>(cond->buffer());
        const auto input1_ptr = reinterpret_cast<const ScalarType *>(in1->buffer());
        const auto input2_ptr = reinterpret_cast<const ScalarType *>(in2->buffer());

        const int outer_size = cond->info()->total_size() / cond->info()->element_size();
        const int inner_size = in1->info()->total_size() / in1->info()->element_size() / outer_size;
        int offset = 0;
        const int step = 16 / in1->info()->element_size();

        for (int i = 0; i < outer_size; ++i) {
            int x = offset;
            const auto input_ptr = static_cast<bool>(*(condition_ptr + i)) ? input1_ptr : input2_ptr;
            for (; x <= offset + inner_size - step; x += step) {
                wrapper::vstore(output_ptr + x, wrapper::vloadq(input_ptr + x));
            }
            if (x <= offset + inner_size - step / 2) {
                wrapper::vstore(output_ptr + x, wrapper::vload(input_ptr + x));
                x += step / 2;
            }
            for (; x < offset + inner_size; ++x) {
                *(output_ptr + x) = *(input_ptr + x);
            }
            offset += inner_size;
        }
    }
}
