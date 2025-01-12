//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_NEON_KERNELS_ADD_IMPL_HPP
#define BATMANINFER_NEON_KERNELS_ADD_IMPL_HPP

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_traits.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/neon/wrapper/wrapper.hpp>

namespace BatmanInfer {
    namespace cpu {
        template<typename ScalarType>
        void add_same_neon(const BIITensor *src0,
                           const BIITensor *src1,
                           BIITensor *dst,
                           const BIConvertPolicy &policy,
                           const BIWindow &window) {
            /** SIMD vector tag type. */
            using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<ScalarType, wrapper::traits::BitWidth::W128>;
            // Create input windows
            BIWindow input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
            BIWindow input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

            // Clear X Dimension on execution window as we handle manually
            BIWindow win = window;
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            constexpr int window_step_x = 16 / sizeof(ScalarType);
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());
            const bool is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

            if (is_broadcast_across_x) {
                const bool is_broadcast_input_2 = input2_win.x().step() == 0;
                BIWindow broadcast_win = is_broadcast_input_2 ? input2_win : input1_win;
                BIWindow non_broadcast_win = !is_broadcast_input_2 ? input2_win : input1_win;
                const BIITensor *broadcast_tensor = is_broadcast_input_2 ? src1 : src0;
                const BIITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

                // Clear X Dimension on execution window as we handle manually
                non_broadcast_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator broadcast_input(broadcast_tensor, broadcast_win);
                BIIterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
                BIIterator output(dst, win);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto non_broadcast_input_ptr = reinterpret_cast<const ScalarType *>(non_broadcast_input.ptr());
                            const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

                            const ScalarType broadcast_value = *reinterpret_cast<const ScalarType *>(broadcast_input.ptr());
                            const auto broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});

                            // Compute S elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                                const auto res = (policy == BIConvertPolicy::SATURATE)
                                                 ? wrapper::vqadd(broadcast_value_vec, non_broadcast_v)
                                                 : wrapper::vadd(broadcast_value_vec, non_broadcast_v);
                                wrapper::vstore(output_ptr + x, res);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                                *(output_ptr + x) = (policy == BIConvertPolicy::SATURATE)
                                                    ? wrapper::add_sat(broadcast_value, non_broadcast_v)
                                                    : broadcast_value + non_broadcast_v;
                            }
                        },
                        broadcast_input, non_broadcast_input, output);
            } else {
                // Clear X Dimension on execution window as we handle manually
                input1_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                input2_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator input1(src0, input1_win);
                BIIterator input2(src1, input2_win);
                BIIterator output(dst, win);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
                            const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());
                            const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

                            // Compute S elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const auto val1 = wrapper::vloadq(input1_ptr + x);
                                const auto val2 = wrapper::vloadq(input2_ptr + x);
                                const auto res =
                                        (policy == BIConvertPolicy::SATURATE) ? wrapper::vqadd(val1, val2)
                                                                              : wrapper::vadd(val1, val2);
                                wrapper::vstore(output_ptr + x, res);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const auto val1 = *(input1_ptr + x);
                                const auto val2 = *(input2_ptr + x);
                                *(output_ptr + x) =
                                        (policy == BIConvertPolicy::SATURATE) ? wrapper::add_sat(val1, val2) : val1 +
                                                                                                               val2;
                            }
                        },
                        input1, input2, output);
            }
        }

        bool
        add_q8_neon_fixedpoint_possible(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);

        bool
        sub_q8_neon_fixedpoint_possible(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);

        bool add_sub_q8_neon_fixedpoint_possible(const BIITensorInfo *src0,
                                                 const BIITensorInfo *src1,
                                                 const BIITensorInfo *dst,
                                                 bool is_addition);

        void add_sub_qasymm8_neon(const BIITensor *src0,
                                  const BIITensor *src1,
                                  BIITensor *dst,
                                  const BIConvertPolicy &policy,
                                  const BIWindow &window,
                                  bool is_addition);

        void add_sub_qasymm8_signed_neon(const BIITensor *src0,
                                         const BIITensor *src1,
                                         BIITensor *dst,
                                         const BIConvertPolicy &policy,
                                         const BIWindow &window,
                                         bool is_addition);

        template<typename ScalarType>
        void add_q8_neon_fixedpoint(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window);

        template<typename ScalarType>
        void add_sub_q8_neon_fixedpoint(const BIITensor *src0,
                                        const BIITensor *src1,
                                        BIITensor *dst,
                                        const BIConvertPolicy &policy,
                                        const BIWindow &window,
                                        bool is_addition);
    }
}

#endif //BATMANINFER_NEON_KERNELS_ADD_IMPL_HPP
