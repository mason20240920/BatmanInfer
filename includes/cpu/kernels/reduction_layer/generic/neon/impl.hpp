//
// Created by Mason on 2025/4/10.
//

#pragma once

#include <data/core/bi_coordinates.hpp>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_tensor_info.hpp>

#include <data/core/neon/bi_neon_math.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>
#include <support/bi_saturate_cast.hpp>

#include <arm_neon.h>

namespace BatmanInfer {
    // Helper function that calls vqmovun/vqmvn, vcombine and vstore, allows templating of RedOpYZW_quantized
    template<typename T>
    void combine_and_store(int16x8_t t1, int16x8_t t2, BIIterator &output, int offset = 0) {
        if (std::is_same<T, uint8_t>::value) {
            auto res = wrapper::vcombine(wrapper::vqmovun(t1), wrapper::vqmovun(t2));
            wrapper::vstore(output.ptr() + offset, res);
        } else {
            auto res = wrapper::vcombine(wrapper::vqmovn(t1), wrapper::vqmovn(t2));
            wrapper::vstore(reinterpret_cast<int8_t *>(output.ptr() + offset), res);
        }
    }

    template<typename T>
    uint32x4x4_t calculate_index(uint32_t idx, T a, T b, uint32x4x4_t c, BIReductionOperation op, int axis) {
        uint32x4_t mask{0};
        if (op == BIReductionOperation::ARG_IDX_MIN) {
            mask = wrapper::vcgt(b, a);
        } else {
            mask = wrapper::vclt(b, a);
        }

        uint32x4_t vec_idx = {idx, idx + 1, idx + 2, idx + 3};
        if (axis != 0) {
            vec_idx = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        }
        uint32x4x4_t res = {{wrapper::vbsl(mask, vec_idx, c.val[0]), 0, 0, 0}};

        return res;
    }

    template<typename T>
    uint32x4x4_t calculate_index_quantized(uint32_t idx, T a, T b, uint32x4x4_t c, BIReductionOperation op, int axis) {
        uint32x4x4_t mask{{0}};
        uint8x16_t mask_u8{0};
        if (op == BIReductionOperation::ARG_IDX_MIN) {
            mask_u8 = wrapper::vcgt(b, a);
        } else {
            mask_u8 = wrapper::vclt(b, a);
        }
        auto wide_u16_1 =
                wrapper::vorr(vshll_n_u8(wrapper::vgetlow(mask_u8), 8), wrapper::vmovl(wrapper::vgetlow(mask_u8)));
        auto wide_u16_2 =
                wrapper::vorr(vshll_n_u8(wrapper::vgethigh(mask_u8), 8), wrapper::vmovl(wrapper::vgethigh(mask_u8)));
        mask.val[0] =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgetlow(wide_u16_1), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_1)));
        mask.val[1] =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgethigh(wide_u16_1), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_1)));
        mask.val[2] =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgetlow(wide_u16_2), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_2)));
        mask.val[3] =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgethigh(wide_u16_2), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_2)));

        uint32x4x4_t vec_idx = {
            {
                {idx + 0, idx + 1, idx + 2, idx + 3},
                {idx + 4, idx + 5, idx + 6, idx + 7},
                {idx + 8, idx + 9, idx + 10, idx + 11},
                {idx + 12, idx + 13, idx + 14, idx + 15}
            }
        };
        if (axis != 0) {
            vec_idx.val[0] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
            vec_idx.val[1] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
            vec_idx.val[2] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
            vec_idx.val[3] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        }
        uint32x4x4_t res = {
            {
                vbslq_u32(mask.val[0], vec_idx.val[0], c.val[0]), vbslq_u32(mask.val[1], vec_idx.val[1], c.val[1]),
                vbslq_u32(mask.val[2], vec_idx.val[2], c.val[2]), vbslq_u32(mask.val[3], vec_idx.val[3], c.val[3])
            }
        };

        return res;
    }

    // Helper function to calculate the minimum value of the input vector. All the elements in the output vector contain the min value.
    template<typename T>
    inline typename std::enable_if<
        std::is_same<T, float32x4_t>::value || std::is_same<T, int32x4_t>::value,
        typename std::conditional<std::is_same<T, float32x4_t>::value, float32x2_t, int32x2_t>::type>::type
    calculate_min(T in) {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
        return wrapper::vpmin(pmin, pmin);
    }

    // Helper function to calculate the minimum value of the input vector. All the elements in the output vector contain the min value.
    template<typename T>
    inline typename std::enable_if<
        std::is_same<T, uint8x16_t>::value || std::is_same<T, int8x16_t>::value,
        typename std::conditional<std::is_same<T, uint8x16_t>::value, uint8x8_t, int8x8_t>::type>::type
    calculate_min(T in) {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
        pmin = wrapper::vpmin(pmin, pmin);
        pmin = wrapper::vpmin(pmin, pmin);
        return wrapper::vpmin(pmin, pmin);
    }

    // Helper function to calculate the maximum value of the input vector. All the elements in the output vector contain the max value.
    template<typename T>
    inline typename std::enable_if<
        std::is_same<T, float32x4_t>::value || std::is_same<T, int32x4_t>::value,
        typename std::conditional<std::is_same<T, float32x4_t>::value, float32x2_t, int32x2_t>::type>::type
    calculate_max(T in) {
        auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
        return wrapper::vpmax(pmax, pmax);
    }

    // Helper function to calculate the maximum value of the input vector. All the elements in the output vector contain the max value.
    template<typename T>
    inline typename std::enable_if<
        std::is_same<T, uint8x16_t>::value || std::is_same<T, int8x16_t>::value,
        typename std::conditional<std::is_same<T, uint8x16_t>::value, uint8x8_t, int8x8_t>::type>::type
    calculate_max(T in) {
        auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
        pmax = wrapper::vpmax(pmax, pmax);
        pmax = wrapper::vpmax(pmax, pmax);
        return wrapper::vpmax(pmax, pmax);
    }

    template<typename T>
    uint32_t calculate_vector_index(uint32x4x4_t vec_res_idx, T vec_res_value, BIReductionOperation op) {
        uint32x4_t res_idx_mask{0};
        uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);

        if (op == BIReductionOperation::ARG_IDX_MIN) {
            auto pmin = calculate_min(vec_res_value);
            auto mask = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
            res_idx_mask = wrapper::vand(vec_res_idx.val[0], mask);
        } else {
            auto pmax = calculate_max(vec_res_value);
            auto mask = wrapper::vceq(vec_res_value, wrapper::vcombine(pmax, pmax));
            res_idx_mask = wrapper::vand(vec_res_idx.val[0], mask);
        }

        res_idx_mask = wrapper::vadd(res_idx_mask, mask_ones);
        auto pmin = wrapper::vpmin(wrapper::vgethigh(res_idx_mask), wrapper::vgetlow(res_idx_mask));
        pmin = wrapper::vpmin(pmin, pmin);
        uint32_t res = wrapper::vgetlane(pmin, 0);

        return (res - 0xFFFFFFFF);
    }

    template<typename T>
    uint32_t calculate_vector_index_quantized(uint32x4x4_t vec_res_idx, T vec_res_value, BIReductionOperation op) {
        uint32x4x4_t res_idx_mask{{0}};
        uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);
        uint8x16_t mask_u8{0};
        if (op == BIReductionOperation::ARG_IDX_MIN) {
            auto pmin = calculate_min(vec_res_value);
            mask_u8 = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
        } else {
            auto pmax = calculate_max(vec_res_value);
            mask_u8 = wrapper::vceq(vec_res_value, wrapper::vcombine(pmax, pmax));
        }

        // Widen vectors
        auto wide_u16_1 =
                wrapper::vorr(vshll_n_u8(wrapper::vgetlow(mask_u8), 8), wrapper::vmovl(wrapper::vgetlow(mask_u8)));
        auto wide_u16_2 =
                wrapper::vorr(vshll_n_u8(wrapper::vgethigh(mask_u8), 8), wrapper::vmovl(wrapper::vgethigh(mask_u8)));
        auto wide_u32_1 =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgetlow(wide_u16_1), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_1)));
        auto wide_u32_2 =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgethigh(wide_u16_1), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_1)));
        auto wide_u32_3 =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgetlow(wide_u16_2), 16), wrapper::vmovl(wrapper::vgetlow(wide_u16_2)));
        auto wide_u32_4 =
                wrapper::vorr(
                    vshll_n_u16(wrapper::vgethigh(wide_u16_2), 16), wrapper::vmovl(wrapper::vgethigh(wide_u16_2)));
        res_idx_mask.val[0] = wrapper::vand(vec_res_idx.val[0], wide_u32_1);
        res_idx_mask.val[1] = wrapper::vand(vec_res_idx.val[1], wide_u32_2);
        res_idx_mask.val[2] = wrapper::vand(vec_res_idx.val[2], wide_u32_3);
        res_idx_mask.val[3] = wrapper::vand(vec_res_idx.val[3], wide_u32_4);
        res_idx_mask.val[0] = wrapper::vadd(res_idx_mask.val[0], mask_ones);
        res_idx_mask.val[1] = wrapper::vadd(res_idx_mask.val[1], mask_ones);
        res_idx_mask.val[2] = wrapper::vadd(res_idx_mask.val[2], mask_ones);
        res_idx_mask.val[3] = wrapper::vadd(res_idx_mask.val[3], mask_ones);

        uint32_t res = 0xFFFFFFFF;
        int iter = 0;
        do {
            auto pmin = wrapper::vpmin(wrapper::vgethigh(res_idx_mask.val[iter]),
                                       wrapper::vgetlow(res_idx_mask.val[iter]));
            pmin = wrapper::vpmin(pmin, pmin);
            res = std::min(wrapper::vgetlane(pmin, 0), res);
            iter++;
        } while (iter < 4);

        return (res - 0xFFFFFFFF);
    }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    template<>
    uint32x4x4_t inline calculate_index(
        uint32_t idx, float16x8_t a, float16x8_t b, uint32x4x4_t c, BIReductionOperation op, int axis) {
        uint32x4x2_t mask{0};
        uint16x8_t mask_u16{0};
        if (op == BIReductionOperation::ARG_IDX_MIN) {
            mask_u16 = wrapper::vcgt(b, a);
        } else {
            mask_u16 = wrapper::vclt(b, a);
        }
        mask.val[0] = wrapper::vmovl(wrapper::vgetlow(mask_u16));
        mask.val[1] = wrapper::vmovl(wrapper::vgethigh(mask_u16));
        uint32x4x2_t vec_idx = {{{idx + 0, idx + 1, idx + 2, idx + 3}, {idx + 4, idx + 5, idx + 6, idx + 7}}};
        if (axis != 0) {
            vec_idx.val[0] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
            vec_idx.val[1] = wrapper::vdup_n(idx, wrapper::traits::vector_128_tag{});
        }
        uint32x4x4_t res = {
            wrapper::vbsl(mask.val[0], vec_idx.val[0], c.val[0]),
            wrapper::vbsl(mask.val[1], vec_idx.val[1], c.val[1]), 0, 0
        };

        return res;
    }

    // Helper function to calculate the minimum value of the input vector. All the elements in the output vector contain the min value.
    inline float16x4_t calculate_min(float16x8_t in) {
        auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
        pmin = wrapper::vpmin(pmin, pmin);
        return wrapper::vpmin(pmin, pmin);
    }

    // Helper function to calculate the maximum value of the input vector. All the elements in the output vector contain the max value.
    inline float16x4_t calculate_max(float16x8_t in) {
        auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
        pmax = wrapper::vpmax(pmax, pmax);
        return wrapper::vpmax(pmax, pmax);
    }

    template<>
    inline uint32_t calculate_vector_index(uint32x4x4_t vec_res_idx, float16x8_t vec_res_value,
                                           BIReductionOperation op) {
        uint32x4x2_t res_idx_mask{0};
        uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);
        uint16x8_t mask_u16;
        if (op == BIReductionOperation::ARG_IDX_MIN) {
            auto pmin = calculate_min(vec_res_value);
            mask_u16 = wrapper::vceq(vec_res_value, wrapper::vcombine(pmin, pmin));
        } else {
            auto pmax = calculate_max(vec_res_value);
            mask_u16 = wrapper::vceq(vec_res_value, wrapper::vcombine(pmax, pmax));
        }

        // Widen vectors
        auto wide_u32_1 =
                wrapper::vorr(vshll_n_u16(wrapper::vgetlow(mask_u16), 8), wrapper::vmovl(wrapper::vgetlow(mask_u16)));
        auto wide_u32_2 =
                wrapper::vorr(vshll_n_u16(wrapper::vgethigh(mask_u16), 8), wrapper::vmovl(wrapper::vgethigh(mask_u16)));
        res_idx_mask.val[0] = wrapper::vand(vec_res_idx.val[0], wide_u32_1);
        res_idx_mask.val[1] = wrapper::vand(vec_res_idx.val[1], wide_u32_2);
        res_idx_mask.val[0] = wrapper::vadd(res_idx_mask.val[0], mask_ones);
        res_idx_mask.val[1] = wrapper::vadd(res_idx_mask.val[1], mask_ones);

        uint32_t res = 0xFFFFFFFF;
        uint32_t iter = 0;
        do {
            auto pmin = wrapper::vpmin(wrapper::vgethigh(res_idx_mask.val[iter]),
                                       wrapper::vgetlow(res_idx_mask.val[iter]));
            pmin = wrapper::vpmin(pmin, pmin);
            res = std::min(wrapper::vgetlane(pmin, 0), res);
            iter++;
        } while (iter < 2);

        return (res - 0xFFFFFFFF);
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    template<class F>
    class Reducer {
    public:
        static void reduceX(const BIWindow &window, const BIITensor *input, BIITensor *output, F f,
                            const BIReductionOperation op) {
            // Set out window
            BIWindow out_window(window);
            out_window.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            f(window, out_window, input, output, op);
        }

        static void reduceY(const BIWindow &window, const BIITensor *input, BIITensor *output, F f,
                            const BIReductionOperation op) {
            // Set in window
            BIWindow in_window(window);
            BIWindow out_window(window);

            in_window.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));
            out_window.set(BIWindow::DimY,
                           BIWindow::BIDimension(0, output->info()->dimension(1), output->info()->dimension(1)));

            f(in_window, out_window, input, output, 1, op);
        }

        static void reduceZ(const BIWindow &window, const BIITensor *input, BIITensor *output, F f,
                            const BIReductionOperation op) {
            // Set in window
            BIWindow in_window(window);
            BIWindow out_window(window);

            in_window.set(BIWindow::DimZ, BIWindow::BIDimension(0, 1, 1));
            out_window.set(BIWindow::DimZ,
                           BIWindow::BIDimension(0, output->info()->dimension(2), output->info()->dimension(2)));

            f(in_window, out_window, input, output, 2, op);
        }

        static void reduceW(const BIWindow &window, const BIITensor *input, BIITensor *output, F f,
                            const BIReductionOperation op) {
            // Set in/out window
            BIWindow in_window(window);
            BIWindow out_window(window);

            in_window.set(3, BIWindow::BIDimension(0, 1, 1));
            out_window.set(3, BIWindow::BIDimension(0, 1, 1));

            f(in_window, out_window, input, output, 3, op);
        }
    };

    template<typename T, int S>
    struct RedOpX {
        /** SIMD vector tag type. */
        using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

        inline void operator()(
            const BIWindow &in_window, BIWindow &out_window, const BIITensor *in, BIITensor *out,
            const BIReductionOperation op) {
            const size_t input_dim_0 = in->info()->dimension(0);
            const int window_step_x = 16 / sizeof(T);
            const auto window_start_x = static_cast<int>(in_window.x().start());
            const auto window_end_x = static_cast<int>(in_window.x().end());

            BIWindow in_win_no_pad = in_window;
            in_win_no_pad.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(in, in_win_no_pad);
            BIIterator output(out, out_window);

            execute_window_loop(
                in_win_no_pad,
                [&](const BICoordinates &) {
                    const auto input_ptr = reinterpret_cast<const T *>(input.ptr());

                    auto init_res_value = static_cast<T>(0.f);
                    switch (op) {
                        case BIReductionOperation::ARG_IDX_MAX:
                        case BIReductionOperation::ARG_IDX_MIN:
                        case BIReductionOperation::MIN:
                        case BIReductionOperation::MAX: {
                            init_res_value = static_cast<T>(*input_ptr);
                            break;
                        }
                        case BIReductionOperation::PROD: {
                            init_res_value = static_cast<T>(1.f);
                            break;
                        }
                        default:
                            break;
                    }
                    auto vec_res_value = wrapper::vdup_n(init_res_value, ExactTagType{});
                    uint32x4x4_t vec_res_idx{{0}};

                    // Compute window_step_x elements per iteration
                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        const auto vec_elements = wrapper::vloadq(input_ptr + x);
                        switch (op) {
                            case BIReductionOperation::SUM_SQUARE:
                                vec_res_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements), vec_res_value);
                                break;
                            case BIReductionOperation::MEAN_SUM:
                            case BIReductionOperation::SUM:
                                vec_res_value = wrapper::vadd(vec_elements, vec_res_value);
                                break;
                            case BIReductionOperation::PROD:
                                vec_res_value = wrapper::vmul(vec_elements, vec_res_value);
                                break;
                            case BIReductionOperation::ARG_IDX_MIN: {
                                auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                vec_res_idx = calculate_index<decltype(vec_res_value)>(
                                    x, temp_vec_res_value, vec_res_value,
                                    vec_res_idx, op, 0);
                                vec_res_value = temp_vec_res_value;
                                break;
                            }
                            case BIReductionOperation::ARG_IDX_MAX: {
                                auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                vec_res_idx = calculate_index<decltype(vec_res_value)>(
                                    x, temp_vec_res_value, vec_res_value,
                                    vec_res_idx, op, 0);
                                vec_res_value = temp_vec_res_value;
                                break;
                            }
                            case BIReductionOperation::MIN: {
                                vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                break;
                            }
                            case BIReductionOperation::MAX: {
                                vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                break;
                            }
                            default:
                                BI_COMPUTE_ERROR("Not supported");
                        }
                    }

                    switch (op) {
                        case BIReductionOperation::SUM:
                        case BIReductionOperation::MEAN_SUM:
                        case BIReductionOperation::SUM_SQUARE: {
#ifdef BI_COMPUTE_DEBUG_ENABLED
                            auto res = static_cast<T>(0.f);
                            for (int i = 0; i < S; ++i) {
                                res += wrapper::vgetlane(vec_res_value, i);
                            }
#else  // BI_COMPUTE_DEBUG_ENABLED
                            auto carry_res =
                                    wrapper::vpadd(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
                            for (int i = 0; i < S / 4; ++i) {
                                carry_res = wrapper::vpadd(carry_res, carry_res);
                            }
                            auto res = wrapper::vgetlane(carry_res, 0);
#endif // BI_COMPUTE_DEBUG_ENABLED
                            if (op == BIReductionOperation::SUM_SQUARE) {
                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    res += (*(input_ptr + x)) * (*(input_ptr + x));
                                }
                            } else {
                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    res += *(input_ptr + x);
                                }
                            }

                            if (op == BIReductionOperation::MEAN_SUM) {
                                res /= input_dim_0;
                            }

                            *(reinterpret_cast<T *>(output.ptr())) = res;
                            break;
                        }
                        case BIReductionOperation::PROD: {
                            auto carry_res =
                                    wrapper::vmul(wrapper::vgethigh(vec_res_value), wrapper::vgetlow(vec_res_value));
                            T res = 1;
                            for (int i = 0; i < S / 2; ++i) {
                                res *= wrapper::vgetlane(carry_res, i);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                res *= *(input_ptr + x);
                            }

                            *(reinterpret_cast<T *>(output.ptr())) = res;
                            break;
                        }
                        case BIReductionOperation::ARG_IDX_MIN: {
                            auto idx = calculate_vector_index<decltype(vec_res_value)>(vec_res_idx, vec_res_value, op);
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_min(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                if (*(input_ptr + x) < res) {
                                    idx = x;
                                    res = *(input_ptr + x);
                                }
                            }
                            *(reinterpret_cast<uint32_t *>(output.ptr())) = idx;
                            break;
                        }
                        case BIReductionOperation::ARG_IDX_MAX: {
                            auto idx = calculate_vector_index<decltype(vec_res_value)>(vec_res_idx, vec_res_value, op);
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_max(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                if (*(input_ptr + x) > res) {
                                    idx = x;
                                    res = *(input_ptr + x);
                                }
                            }
                            *(reinterpret_cast<uint32_t *>(output.ptr())) = idx;
                            break;
                        }
                        case BIReductionOperation::MIN: {
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_min(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                res = *(input_ptr + x) < res ? *(input_ptr + x) : res;
                            }
                            *(reinterpret_cast<T *>(output.ptr())) = res;
                            break;
                        }
                        case BIReductionOperation::MAX: {
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_max(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                res = *(input_ptr + x) > res ? *(input_ptr + x) : res;
                            }
                            *(reinterpret_cast<T *>(output.ptr())) = res;
                            break;
                        }
                        default:
                            BI_COMPUTE_ERROR("Not supported");
                    }
                },
                input, output);
        }
    };

    template<typename T>
    struct RedOpX_quantized {
        inline void operator()(
            const BIWindow &in_window, BIWindow &out_window, const BIITensor *in, BIITensor *out,
            const BIReductionOperation op) {
            using PromotedType = typename wrapper::traits::promote<typename wrapper::traits::promote<T>::type>::type;

            const auto oq_info = out->info()->quantization_info().uniform();

            const BITensorInfo in_info = *(in->info());
            const BIUniformQuantizationInfo iq_info = in_info.quantization_info().uniform();

            const int window_step_x = 16 / sizeof(T);
            const auto window_start_x = static_cast<int>(in_window.x().start());
            const auto window_end_x = static_cast<int>(in_window.x().end());

            BIWindow in_win_no_pad = in_window;
            in_win_no_pad.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator input(in, in_win_no_pad);
            BIIterator output(out, out_window);

            const auto in_offset = static_cast<float>(iq_info.offset);
            const float in_scale = iq_info.scale;

            const auto out_offset = static_cast<float>(oq_info.offset);
            const float out_scale = oq_info.scale;

            const auto num_elements = static_cast<float>(in_info.dimension(0));

            const float A = in_scale / (out_scale * num_elements);
            const float B = out_offset - (in_scale * in_offset) / (out_scale);

            execute_window_loop(
                in_win_no_pad,
                [&](const BICoordinates &) {
                    const auto input_ptr = reinterpret_cast<T *>(input.ptr());

                    auto vec_res_value1 =
                            wrapper::vdup_n(static_cast<PromotedType>(0.f), wrapper::traits::vector_128_tag{});
                    auto vec_res_value2 =
                            wrapper::vdup_n(static_cast<PromotedType>(0.f), wrapper::traits::vector_128_tag{});
                    auto vec_res_value3 =
                            wrapper::vdup_n(static_cast<PromotedType>(0.f), wrapper::traits::vector_128_tag{});
                    auto vec_res_value4 =
                            wrapper::vdup_n(static_cast<PromotedType>(0.f), wrapper::traits::vector_128_tag{});

                    auto vec_res_value1_f = vdupq_n_f32(static_cast<float>(1.f));
                    auto vec_res_value2_f = vdupq_n_f32(static_cast<float>(1.f));
                    auto vec_res_value3_f = vdupq_n_f32(static_cast<float>(1.f));
                    auto vec_res_value4_f = vdupq_n_f32(static_cast<float>(1.f));

                    typename wrapper::traits::neon_vector<T, 16>::type vec_res_value = {0};

                    if (op == BIReductionOperation::ARG_IDX_MAX || op == BIReductionOperation::ARG_IDX_MIN ||
                        op == BIReductionOperation::MIN || op == BIReductionOperation::MAX) {
                        vec_res_value = wrapper::vdup_n(*input_ptr, wrapper::traits::vector_128_tag{});
                    }

                    uint32x4x4_t vec_res_idx{{0}};
                    // Compute window_step_x elements per iteration
                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        const auto vec_elements = wrapper::vloadq(input_ptr + x);
                        switch (op) {
                            case BIReductionOperation::SUM:
                            case BIReductionOperation::MEAN_SUM: {
                                const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                                const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                                const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                                const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                                const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                                const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                                vec_res_value1 = wrapper::vadd(temp32x4t_1, vec_res_value1);
                                vec_res_value2 = wrapper::vadd(temp32x4t_2, vec_res_value2);
                                vec_res_value3 = wrapper::vadd(temp32x4t_3, vec_res_value3);
                                vec_res_value4 = wrapper::vadd(temp32x4t_4, vec_res_value4);
                                break;
                            }
                            case BIReductionOperation::PROD: {
                                const auto offset32x4f_4 = vdupq_n_f32(iq_info.offset);
                                const auto scale32x4f_4 = vdupq_n_f32(iq_info.scale);

                                const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                                const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                                const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                                const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                                const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                                const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                                auto temp32x4f_1 = wrapper::vcvt<float>(temp32x4t_1);
                                auto temp32x4f_2 = wrapper::vcvt<float>(temp32x4t_2);
                                auto temp32x4f_3 = wrapper::vcvt<float>(temp32x4t_3);
                                auto temp32x4f_4 = wrapper::vcvt<float>(temp32x4t_4);

                                //de-quantize vec_elements
                                temp32x4f_1 = vmulq_f32(vsubq_f32(temp32x4f_1, offset32x4f_4), scale32x4f_4);
                                temp32x4f_2 = vmulq_f32(vsubq_f32(temp32x4f_2, offset32x4f_4), scale32x4f_4);
                                temp32x4f_3 = vmulq_f32(vsubq_f32(temp32x4f_3, offset32x4f_4), scale32x4f_4);
                                temp32x4f_4 = vmulq_f32(vsubq_f32(temp32x4f_4, offset32x4f_4), scale32x4f_4);

                                vec_res_value1_f = vmulq_f32(temp32x4f_1, vec_res_value1_f);
                                vec_res_value2_f = vmulq_f32(temp32x4f_2, vec_res_value2_f);
                                vec_res_value3_f = vmulq_f32(temp32x4f_3, vec_res_value3_f);
                                vec_res_value4_f = vmulq_f32(temp32x4f_4, vec_res_value4_f);
                                break;
                            }
                            case BIReductionOperation::ARG_IDX_MIN: {
                                auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                vec_res_idx = calculate_index_quantized<decltype(vec_res_value)>(
                                    x, temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                                vec_res_value = temp_vec_res_value;
                                break;
                            }
                            case BIReductionOperation::ARG_IDX_MAX: {
                                auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                vec_res_idx = calculate_index_quantized<decltype(vec_res_value)>(
                                    x, temp_vec_res_value, vec_res_value, vec_res_idx, op, 0);
                                vec_res_value = temp_vec_res_value;
                                break;
                            }
                            case BIReductionOperation::MIN: {
                                vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                break;
                            }
                            case BIReductionOperation::MAX: {
                                vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                break;
                            }
                            default:
                                BI_COMPUTE_ERROR("Not supported");
                        }
                    }

                    switch (op) {
                        case BIReductionOperation::ARG_IDX_MIN: {
                            auto idx =
                                    calculate_vector_index_quantized<decltype(vec_res_value)>(
                                        vec_res_idx, vec_res_value, op);
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_min(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                if (*(input_ptr + x) < res) {
                                    idx = x;
                                    res = *(input_ptr + x);
                                }
                            }
                            *(reinterpret_cast<uint32_t *>(output.ptr())) = idx;
                            break;
                        }
                        case BIReductionOperation::ARG_IDX_MAX: {
                            auto idx =
                                    calculate_vector_index_quantized<decltype(vec_res_value)>(
                                        vec_res_idx, vec_res_value, op);
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_max(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                if (*(input_ptr + x) > res) {
                                    idx = x;
                                    res = *(input_ptr + x);
                                }
                            }
                            *(reinterpret_cast<uint32_t *>(output.ptr())) = idx;
                            break;
                        }
                        case BIReductionOperation::MIN: {
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_min(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                res = *(input_ptr + x) < res ? *(input_ptr + x) : res;
                            }
                            *(reinterpret_cast<T *>(output.ptr())) = res;
                            break;
                        }
                        case BIReductionOperation::MAX: {
                            auto res = static_cast<T>(wrapper::vgetlane(calculate_max(vec_res_value), 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                res = *(input_ptr + x) > res ? *(input_ptr + x) : res;
                            }
                            *(reinterpret_cast<T *>(output.ptr())) = res;
                            break;
                        }
                        case BIReductionOperation::PROD: {
                            auto carry_res = wrapper::vmul(vec_res_value1_f, vec_res_value2_f);
                            carry_res = wrapper::vmul(carry_res, vec_res_value3_f);
                            carry_res = wrapper::vmul(carry_res, vec_res_value4_f);

                            float res = wrapper::vgetlane(carry_res, 0);
                            res *= wrapper::vgetlane(carry_res, 1);
                            res *= wrapper::vgetlane(carry_res, 2);
                            res *= wrapper::vgetlane(carry_res, 3);

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                //de-quantize input
                                if (std::is_same<T, uint8_t>::value) {
                                    res *= dequantize_qasymm8(*(input_ptr + x), iq_info);
                                } else {
                                    res *= dequantize_qasymm8_signed(*(input_ptr + x), iq_info);
                                }
                            }

                            //re-quantize result
                            if (std::is_same<T, uint8_t>::value) {
                                res = quantize_qasymm8(res, iq_info);
                            } else {
                                res = quantize_qasymm8_signed(res, iq_info);
                            }

                            *reinterpret_cast<T *>(output.ptr()) = static_cast<T>(res);
                            break;
                        }
                        case BIReductionOperation::SUM:
                        case BIReductionOperation::MEAN_SUM: {
                            auto carry_res = wrapper::vadd(vec_res_value1, vec_res_value2);
                            carry_res = wrapper::vadd(carry_res, vec_res_value3);
                            carry_res = wrapper::vadd(carry_res, vec_res_value4);

                            auto carry_paddition =
                                    wrapper::vpadd(wrapper::vgethigh(carry_res), wrapper::vgetlow(carry_res));
                            carry_paddition = wrapper::vpadd(carry_paddition, carry_paddition);
                            auto res = static_cast<int32_t>(wrapper::vgetlane(carry_paddition, 0));

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                res += *(input_ptr + x);
                            }

                            if (op == BIReductionOperation::MEAN_SUM) {
                                const float resFinal = A * (static_cast<float>(res)) + B;
                                *reinterpret_cast<T *>(output.ptr()) = utils::cast::saturate_cast<T>(resFinal);
                            } else {
                                // Subtract accumulated offsets
                                res -= (in_info.dimension(0) - 1) * iq_info.offset;
                                *reinterpret_cast<T *>(output.ptr()) = utils::cast::saturate_cast<T>(res);
                            }

                            break;
                        }
                        default:
                            BI_COMPUTE_ERROR("Not supported");
                    }
                },
                input, output);
        }
    };

    template<typename T, int S>
    struct RedOpYZW {
        /** SIMD vector tag type. */
        using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
        using neon_vector = typename wrapper::traits::neon_vector<T, S>::type;

        inline void operator()(const BIWindow &in_window,
                               BIWindow &out_window,
                               const BIITensor *in,
                               BIITensor *out,
                               int axis,
                               const BIReductionOperation op) {
            const BITensorInfo in_info = *(in->info());
            const int window_step_x = 16 / sizeof(T);
            const auto window_start_x_tmp = static_cast<int>(in_window.x().start());
            const auto window_end_x_tmp = static_cast<int>(in_window.x().end());
            // As it split over x-axis, need to set the correct spiltted window start and end.
            const auto window_start_x = static_cast<int>(0);
            const auto window_end_x = static_cast<int>(in_window.shape().x());

            BIWindow in_win_no_pad = in_window;
            in_win_no_pad.set(BIWindow::DimX,
                              BIWindow::BIDimension(window_start_x_tmp, window_end_x_tmp, in_window.shape().x()));
            BIWindow out_win_no_pad = out_window;
            out_win_no_pad.set(BIWindow::DimX,
                               BIWindow::BIDimension(window_start_x_tmp, window_end_x_tmp, out_window.shape().x()));

            BIIterator input(in, in_win_no_pad);
            BIIterator output(out, out_win_no_pad);

            execute_window_loop(
                in_win_no_pad,
                [&](const BICoordinates &) {
                    const auto input_ptr = reinterpret_cast<T *>(input.ptr());

                    // Compute window_step_x elements per iteration
                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        neon_vector vec_res_value = {0};
                        switch (op) {
                            case BIReductionOperation::ARG_IDX_MAX:
                            case BIReductionOperation::ARG_IDX_MIN:
                            case BIReductionOperation::MIN:
                            case BIReductionOperation::MAX: {
                                vec_res_value = wrapper::vloadq(input_ptr + x);
                                break;
                            }
                            case BIReductionOperation::PROD: {
                                vec_res_value = wrapper::vdup_n(static_cast<T>(1.f), ExactTagType{});
                                break;
                            }
                            default: {
                                vec_res_value = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
                                break;
                            }
                        }
                        uint32x4x4_t vec_res_idx{{0}};

                        for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim) {
                            const T *in_ptr =
                                    reinterpret_cast<T *>(
                                        input.ptr() + x * sizeof(T) + in_info.strides_in_bytes()[axis] * dim);
                            const auto vec_elements = wrapper::vloadq(in_ptr);
                            switch (op) {
                                case BIReductionOperation::SUM:
                                case BIReductionOperation::MEAN_SUM:
                                    vec_res_value = wrapper::vadd(vec_elements, vec_res_value);
                                    break;
                                case BIReductionOperation::SUM_SQUARE:
                                    vec_res_value = wrapper::vadd(wrapper::vmul(vec_elements, vec_elements),
                                                                  vec_res_value);
                                    break;
                                case BIReductionOperation::PROD:
                                    vec_res_value = wrapper::vmul(vec_elements, vec_res_value);
                                    break;
                                case BIReductionOperation::ARG_IDX_MIN: {
                                    auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                    vec_res_idx =
                                            calculate_index(dim, temp_vec_res_value, vec_res_value, vec_res_idx, op,
                                                            axis);
                                    vec_res_value = temp_vec_res_value;
                                    break;
                                }
                                case BIReductionOperation::ARG_IDX_MAX: {
                                    auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                    vec_res_idx =
                                            calculate_index(dim, temp_vec_res_value, vec_res_value, vec_res_idx, op,
                                                            axis);
                                    vec_res_value = temp_vec_res_value;
                                    break;
                                }
                                case BIReductionOperation::MIN: {
                                    vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                    break;
                                }
                                case BIReductionOperation::MAX: {
                                    vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                    break;
                                }
                                default:
                                    BI_COMPUTE_ERROR("Not supported");
                            }
                        }

                        if (op == BIReductionOperation::MEAN_SUM) {
                            auto vec_width_inv =
                                    wrapper::vinv(
                                        wrapper::vdup_n(static_cast<T>(in_info.dimension(axis)), ExactTagType{}));
                            vec_res_value = wrapper::vmul(vec_res_value, vec_width_inv);
                        }

                        if (op == BIReductionOperation::ARG_IDX_MIN || op == BIReductionOperation::ARG_IDX_MAX) {
                            wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + x, vec_res_idx.val[0]);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                            if (std::is_same<T, float16_t>::value) {
                                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr()) + x + 4, vec_res_idx.val[1]);
                            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        } else {
                            wrapper::vstore(reinterpret_cast<T *>(output.ptr() + x * sizeof(T)), vec_res_value);
                        }
                    }

                    // Compute left-over elements
                    for (; x < window_end_x; ++x) {
                        auto res_value = 0.f;
                        switch (op) {
                            case BIReductionOperation::ARG_IDX_MAX:
                            case BIReductionOperation::ARG_IDX_MIN:
                            case BIReductionOperation::MIN:
                            case BIReductionOperation::MAX: {
                                res_value = *(input_ptr + x);
                                break;
                            }
                            case BIReductionOperation::PROD: {
                                res_value = static_cast<T>(1.f);
                                break;
                            }
                            default: {
                                res_value = static_cast<T>(0.f);
                                break;
                            }
                        }

                        uint32_t res_idx = 0;
                        for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim) {
                            const T *in_ptr =
                                    reinterpret_cast<T *>(
                                        input.ptr() + x * sizeof(T) + in_info.strides_in_bytes()[axis] * dim);

                            switch (op) {
                                case BIReductionOperation::SUM:
                                case BIReductionOperation::MEAN_SUM:
                                    res_value += *in_ptr;
                                    break;
                                case BIReductionOperation::SUM_SQUARE:
                                    res_value += *in_ptr * *in_ptr;
                                    break;
                                case BIReductionOperation::PROD:
                                    res_value *= *in_ptr;
                                    break;
                                case BIReductionOperation::ARG_IDX_MIN: {
                                    if (*in_ptr < res_value) {
                                        res_value = *in_ptr;
                                        res_idx = dim;
                                    }
                                    break;
                                }
                                case BIReductionOperation::ARG_IDX_MAX: {
                                    if (*in_ptr > res_value) {
                                        res_value = *in_ptr;
                                        res_idx = dim;
                                    }
                                    break;
                                }
                                case BIReductionOperation::MIN: {
                                    res_value = *in_ptr < res_value ? *in_ptr : res_value;
                                    break;
                                }
                                case BIReductionOperation::MAX: {
                                    res_value = *in_ptr > res_value ? *in_ptr : res_value;
                                    break;
                                }
                                default:
                                    BI_COMPUTE_ERROR("Not supported");
                            }
                        }

                        if (op == BIReductionOperation::MEAN_SUM) {
                            res_value /= in_info.dimension(axis);
                        }

                        if (op == BIReductionOperation::ARG_IDX_MIN || op == BIReductionOperation::ARG_IDX_MAX) {
                            *(reinterpret_cast<uint32_t *>(output.ptr()) + x) = res_idx;
                        } else {
                            *(reinterpret_cast<T *>(output.ptr() + x * sizeof(T))) = res_value;
                        }
                    }
                },
                input, output);
        }
    };

    template<typename T, int S, int axis, BIReductionOperation op>
    struct RedOpYZW_complex {
        /** SIMD vector tag type. */
        using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
        using neon_vector = typename wrapper::traits::neon_vector<T, S>::type;

        inline void operator()(
            const BIWindow &in_window, BIWindow &out_window, const BIITensor *in, BIITensor *out, int,
            const BIReductionOperation) {
            BI_COMPUTE_ERROR_ON(axis != 2);
            BI_COMPUTE_ERROR_ON(op != BIReductionOperation::SUM);

            const BITensorInfo in_info = *(in->info());
            const size_t stride_z = in_info.strides_in_bytes()[axis];
            const int window_step_x = 16 / sizeof(T);
            const auto window_start_x_tmp = static_cast<int>(in_window.x().start());
            const auto window_end_x_tmp = static_cast<int>(in_window.x().end());
            // As it split over x-axis, need to set the correct spiltted window start and end.
            const auto window_start_x = static_cast<int>(0);
            const auto window_end_x = static_cast<int>(in_window.shape().x());

            BIWindow in_win_no_pad = in_window;
            in_win_no_pad.set(BIWindow::DimX,
                              BIWindow::BIDimension(window_start_x_tmp, window_end_x_tmp, in_window.shape().x()));
            BIWindow out_win_no_pad = out_window;
            out_win_no_pad.set(BIWindow::DimX,
                               BIWindow::BIDimension(window_start_x_tmp, window_end_x_tmp, out_window.shape().x()));

            BIIterator input(in, in_win_no_pad);
            BIIterator output(out, out_win_no_pad);

            execute_window_loop(
                in_win_no_pad,
                [&](const BICoordinates &) {
                    // Compute window_step_x elements per iteration
                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        neon_vector vec_res_value_0 = {0};
                        neon_vector vec_res_value_1 = {0};

                        vec_res_value_0 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
                        vec_res_value_1 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

                        T *out_ptr = reinterpret_cast<T *>(output.ptr() + 2 * x * sizeof(T));
                        for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim) {
                            T *in_ptr_0 = reinterpret_cast<T *>(input.ptr() + 2 * x * sizeof(T) + stride_z * dim);
                            T *in_ptr_1 = reinterpret_cast<T *>(input.ptr() + 2 * x * sizeof(T) + 16 + stride_z * dim);

                            const auto vec_elements_0 = wrapper::vloadq(in_ptr_0);
                            const auto vec_elements_1 = wrapper::vloadq(in_ptr_1);

                            vec_res_value_0 = wrapper::vadd(vec_elements_0, vec_res_value_0);
                            vec_res_value_1 = wrapper::vadd(vec_elements_1, vec_res_value_1);
                        }

                        wrapper::vstore(out_ptr, vec_res_value_0);
                        wrapper::vstore(out_ptr + 4, vec_res_value_1);
                    }

                    // Compute left-over elements
                    for (; x < window_end_x; ++x) {
                        auto res_value_0 = 0.f;
                        auto res_value_1 = 0.f;

                        T *out_ptr = reinterpret_cast<T *>(output.ptr() + 2 * x * sizeof(T));
                        for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim) {
                            T *in_ptr = reinterpret_cast<T *>(input.ptr() + 2 * x * sizeof(T) + stride_z * dim);
                            res_value_0 += *in_ptr;
                            res_value_1 += *(in_ptr + 1);
                        }
                        *out_ptr = res_value_0;
                        *(out_ptr + 1) = res_value_1;
                    }
                },
                input, output);
        }
    };

    template<typename T>
    struct RedOpYZW_quantized {
        inline void operator()(const BIWindow &in_window,
                               BIWindow &out_window,
                               const BIITensor *in,
                               BIITensor *out,
                               int axis,
                               const BIReductionOperation op) {
            const BITensorInfo in_info = *(in->info());
            const BIUniformQuantizationInfo iq_info = in_info.quantization_info().uniform();
            using PromotedType = typename wrapper::traits::promote<typename wrapper::traits::promote<T>::type>::type;

            const auto oq_info = out->info()->quantization_info().uniform();

            const int window_step_x = 16 / sizeof(T);
            const auto window_start_x_tmp = static_cast<int>(in_window.x().start());
            const auto window_end_x_tmp = static_cast<int>(in_window.x().end());
            // As it split over x-axis, need to set the correct spiltted window start and end.
            const auto window_start_x = static_cast<int>(0);
            const auto window_end_x = static_cast<int>(in_window.shape().x());

            BIWindow in_win_no_pad = in_window;
            in_win_no_pad.set(BIWindow::DimX,
                              BIWindow::BIDimension(window_start_x_tmp, window_end_x_tmp, in_window.shape().x()));
            BIWindow out_win_no_pad = out_window;
            out_win_no_pad.set(BIWindow::DimX,
                               BIWindow::BIDimension(window_start_x_tmp, window_end_x_tmp, out_window.shape().x()));

            BIIterator input(in, in_win_no_pad);
            BIIterator output(out, out_win_no_pad);

            using vector_type =
                    typename wrapper::traits::neon_bitvector<PromotedType, wrapper::traits::BitWidth::W128>::type;
            using vector_type_f = typename wrapper::traits::neon_vector<float, 4>::type;

            vector_type vec_res_value1{};
            vector_type vec_res_value2{};
            vector_type vec_res_value3{};
            vector_type vec_res_value4{};

            vector_type_f vec_res_value1_f{};
            vector_type_f vec_res_value2_f{};
            vector_type_f vec_res_value3_f{};
            vector_type_f vec_res_value4_f{};

            const float in_offset = static_cast<float>(iq_info.offset);
            const float in_scale = iq_info.scale;

            const float out_offset = static_cast<float>(oq_info.offset);
            const float out_scale = oq_info.scale;

            const float num_elements = static_cast<float>(in_info.dimension(axis));

            const float A = in_scale / (out_scale * num_elements);
            const float B = out_offset - (in_scale * in_offset) / (out_scale);

            const auto vec_A = wrapper::vdup_n(static_cast<float>(A), wrapper::traits::vector_128_tag{});
            const auto vec_B = wrapper::vdup_n(static_cast<float>(B), wrapper::traits::vector_128_tag{});

            execute_window_loop(
                in_win_no_pad,
                [&](const BICoordinates &) {
                    const auto input_ptr = reinterpret_cast<T *>(input.ptr());

                    // Compute window_step_x elements per iteration
                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        uint32x4x4_t vec_res_idx{{0}};
                        vec_res_value1 = wrapper::vdup_n(static_cast<PromotedType>(0),
                                                         wrapper::traits::vector_128_tag{});
                        vec_res_value2 = wrapper::vdup_n(static_cast<PromotedType>(0),
                                                         wrapper::traits::vector_128_tag{});
                        vec_res_value3 = wrapper::vdup_n(static_cast<PromotedType>(0),
                                                         wrapper::traits::vector_128_tag{});
                        vec_res_value4 = wrapper::vdup_n(static_cast<PromotedType>(0),
                                                         wrapper::traits::vector_128_tag{});

                        vec_res_value1_f = wrapper::vdup_n(static_cast<float>(1), wrapper::traits::vector_128_tag{});
                        vec_res_value2_f = wrapper::vdup_n(static_cast<float>(1), wrapper::traits::vector_128_tag{});
                        vec_res_value3_f = wrapper::vdup_n(static_cast<float>(1), wrapper::traits::vector_128_tag{});
                        vec_res_value4_f = wrapper::vdup_n(static_cast<float>(1), wrapper::traits::vector_128_tag{});

                        auto vec_res_value = wrapper::vloadq(input_ptr + x);

                        for (unsigned int index_dim = 0; index_dim < in_info.dimension(axis); ++index_dim) {
                            const T *in_ptr = input_ptr + x + in_info.strides_in_bytes()[axis] * index_dim;
                            const auto vec_elements = wrapper::vloadq(in_ptr);
                            switch (op) {
                                case BIReductionOperation::SUM:
                                case BIReductionOperation::MEAN_SUM: {
                                    const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                                    const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                                    const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                                    const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                                    const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                                    const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                                    vec_res_value1 = wrapper::vadd(temp32x4t_1, vec_res_value1);
                                    vec_res_value2 = wrapper::vadd(temp32x4t_2, vec_res_value2);
                                    vec_res_value3 = wrapper::vadd(temp32x4t_3, vec_res_value3);
                                    vec_res_value4 = wrapper::vadd(temp32x4t_4, vec_res_value4);
                                    break;
                                }
                                case BIReductionOperation::PROD: {
                                    const auto offset32x4f_4 = wrapper::vdup_n(static_cast<float>(iq_info.offset),
                                                                               wrapper::traits::vector_128_tag{});
                                    const auto scale32x4f_4 =
                                            wrapper::vdup_n(iq_info.scale, wrapper::traits::vector_128_tag{});

                                    const auto temp16x8t_1 = wrapper::vmovl(wrapper::vgetlow(vec_elements));
                                    const auto temp16x8t_2 = wrapper::vmovl(wrapper::vgethigh(vec_elements));

                                    const auto temp32x4t_1 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_1));
                                    const auto temp32x4t_2 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_1));
                                    const auto temp32x4t_3 = wrapper::vmovl(wrapper::vgetlow(temp16x8t_2));
                                    const auto temp32x4t_4 = wrapper::vmovl(wrapper::vgethigh(temp16x8t_2));

                                    auto temp32x4f_1 = wrapper::vcvt<float>(temp32x4t_1);
                                    auto temp32x4f_2 = wrapper::vcvt<float>(temp32x4t_2);
                                    auto temp32x4f_3 = wrapper::vcvt<float>(temp32x4t_3);
                                    auto temp32x4f_4 = wrapper::vcvt<float>(temp32x4t_4);

                                    //de-quantize vec_elements
                                    temp32x4f_1 =
                                            wrapper::vmul(wrapper::vsub(temp32x4f_1, offset32x4f_4), scale32x4f_4);
                                    temp32x4f_2 =
                                            wrapper::vmul(wrapper::vsub(temp32x4f_2, offset32x4f_4), scale32x4f_4);
                                    temp32x4f_3 =
                                            wrapper::vmul(wrapper::vsub(temp32x4f_3, offset32x4f_4), scale32x4f_4);
                                    temp32x4f_4 =
                                            wrapper::vmul(wrapper::vsub(temp32x4f_4, offset32x4f_4), scale32x4f_4);

                                    vec_res_value1_f = wrapper::vmul(temp32x4f_1, vec_res_value1_f);
                                    vec_res_value2_f = wrapper::vmul(temp32x4f_2, vec_res_value2_f);
                                    vec_res_value3_f = wrapper::vmul(temp32x4f_3, vec_res_value3_f);
                                    vec_res_value4_f = wrapper::vmul(temp32x4f_4, vec_res_value4_f);
                                    break;
                                }
                                case BIReductionOperation::ARG_IDX_MIN: {
                                    auto temp_vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                    vec_res_idx = calculate_index_quantized(
                                        index_dim, temp_vec_res_value, vec_res_value,
                                        vec_res_idx, op, axis);
                                    vec_res_value = temp_vec_res_value;
                                    break;
                                }
                                case BIReductionOperation::ARG_IDX_MAX: {
                                    auto temp_vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                    vec_res_idx = calculate_index_quantized(
                                        index_dim, temp_vec_res_value, vec_res_value,
                                        vec_res_idx, op, axis);
                                    vec_res_value = temp_vec_res_value;
                                    break;
                                }
                                case BIReductionOperation::MIN: {
                                    vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                                    break;
                                }
                                case BIReductionOperation::MAX: {
                                    vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                                    break;
                                }
                                default:
                                    BI_COMPUTE_ERROR("Not supported");
                            }
                        }

                        switch (op) {
                            case BIReductionOperation::ARG_IDX_MIN:
                            case BIReductionOperation::ARG_IDX_MAX: {
                                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr() + 4 * x), vec_res_idx.val[0]);
                                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr() + 4 * x) + 4,
                                                vec_res_idx.val[1]);
                                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr() + 4 * x) + 8,
                                                vec_res_idx.val[2]);
                                wrapper::vstore(reinterpret_cast<uint32_t *>(output.ptr() + 4 * x) + 12,
                                                vec_res_idx.val[3]);
                                break;
                            }
                            case BIReductionOperation::MIN:
                            case BIReductionOperation::MAX: {
                                wrapper::vstore(reinterpret_cast<T *>(output.ptr() + x), vec_res_value);
                                break;
                            }
                            case BIReductionOperation::SUM: {
                                // Subtract offsets
                                auto offsets = vdupq_n_s32((in_info.dimension(axis) - 1) * iq_info.offset);

                                auto vec_res_s_value1 = wrapper::vreinterpret(vec_res_value1);
                                auto vec_res_s_value2 = wrapper::vreinterpret(vec_res_value2);
                                auto vec_res_s_value3 = wrapper::vreinterpret(vec_res_value3);
                                auto vec_res_s_value4 = wrapper::vreinterpret(vec_res_value4);

                                vec_res_s_value1 = wrapper::vsub(vec_res_s_value1, offsets);
                                vec_res_s_value2 = wrapper::vsub(vec_res_s_value2, offsets);
                                vec_res_s_value3 = wrapper::vsub(vec_res_s_value3, offsets);
                                vec_res_s_value4 = wrapper::vsub(vec_res_s_value4, offsets);

                                const auto temp16x8t_1 =
                                        wrapper::vcombine(wrapper::vqmovn(vec_res_s_value1),
                                                          wrapper::vqmovn(vec_res_s_value2));
                                const auto temp16x8t_2 =
                                        wrapper::vcombine(wrapper::vqmovn(vec_res_s_value3),
                                                          wrapper::vqmovn(vec_res_s_value4));

                                combine_and_store<T>(temp16x8t_1, temp16x8t_2, output, x);
                                break;
                            }
                            case BIReductionOperation::MEAN_SUM: {
                                vec_res_value1_f = wrapper::vmla(vec_B, wrapper::vcvt<float>(vec_res_value1), vec_A);
                                vec_res_value2_f = wrapper::vmla(vec_B, wrapper::vcvt<float>(vec_res_value2), vec_A);
                                vec_res_value3_f = wrapper::vmla(vec_B, wrapper::vcvt<float>(vec_res_value3), vec_A);
                                vec_res_value4_f = wrapper::vmla(vec_B, wrapper::vcvt<float>(vec_res_value4), vec_A);

#ifdef __aarch64__
                                vec_res_value1 = wrapper::vcvtn<PromotedType>(vec_res_value1_f);
                                vec_res_value2 = wrapper::vcvtn<PromotedType>(vec_res_value2_f);
                                vec_res_value3 = wrapper::vcvtn<PromotedType>(vec_res_value3_f);
                                vec_res_value4 = wrapper::vcvtn<PromotedType>(vec_res_value4_f);
#else  // defined(__aarch64__)
                            vec_res_value1    = wrapper::vcvt<PromotedType>(vec_res_value1_f);
                            vec_res_value2    = wrapper::vcvt<PromotedType>(vec_res_value2_f);
                            vec_res_value3    = wrapper::vcvt<PromotedType>(vec_res_value3_f);
                            vec_res_value4    = wrapper::vcvt<PromotedType>(vec_res_value4_f);
#endif // __aarch64__

                                const auto temp16x8t_1 =
                                        wrapper::vcombine(wrapper::vqmovn(vec_res_value1),
                                                          wrapper::vqmovn(vec_res_value2));
                                const auto temp16x8t_2 =
                                        wrapper::vcombine(wrapper::vqmovn(vec_res_value3),
                                                          wrapper::vqmovn(vec_res_value4));
                                auto res = wrapper::vcombine(wrapper::vqmovn(temp16x8t_1),
                                                             wrapper::vqmovn(temp16x8t_2));

                                wrapper::vstore(reinterpret_cast<T *>(output.ptr() + x), res);
                                break;
                            }
                            case BIReductionOperation::PROD: {
                                const auto offset32x4f_4 =
                                        wrapper::vdup_n(static_cast<float>(iq_info.offset),
                                                        wrapper::traits::vector_128_tag{});
                                const auto iscale32x4f_4 = vinvq_f32(vdupq_n_f32(iq_info.scale));

                                //re-quantize
                                vec_res_value1_f =
                                        wrapper::vadd(wrapper::vmul(vec_res_value1_f, iscale32x4f_4), offset32x4f_4);
                                vec_res_value2_f =
                                        wrapper::vadd(wrapper::vmul(vec_res_value2_f, iscale32x4f_4), offset32x4f_4);
                                vec_res_value3_f =
                                        wrapper::vadd(wrapper::vmul(vec_res_value3_f, iscale32x4f_4), offset32x4f_4);
                                vec_res_value4_f =
                                        wrapper::vadd(wrapper::vmul(vec_res_value4_f, iscale32x4f_4), offset32x4f_4);

                                vec_res_value1 = wrapper::vcvt<T>(vec_res_value1_f);
                                vec_res_value2 = wrapper::vcvt<T>(vec_res_value2_f);
                                vec_res_value3 = wrapper::vcvt<T>(vec_res_value3_f);
                                vec_res_value4 = wrapper::vcvt<T>(vec_res_value4_f);

                                const auto temp16x8t_1 =
                                        wrapper::vcombine(wrapper::vqmovn(vec_res_value1),
                                                          wrapper::vqmovn(vec_res_value2));
                                const auto temp16x8t_2 =
                                        wrapper::vcombine(wrapper::vqmovn(vec_res_value3),
                                                          wrapper::vqmovn(vec_res_value4));
                                auto res = wrapper::vcombine(wrapper::vqmovn(temp16x8t_1),
                                                             wrapper::vqmovn(temp16x8t_2));

                                wrapper::vstore(reinterpret_cast<T *>(output.ptr() + x), res);
                                break;
                            }
                            default:
                                BI_COMPUTE_ERROR("Not supported");
                        }
                    }

                    // Compute left-over elements
                    for (; x < window_end_x; ++x) {
                        float res_value = 0.f;
                        int32_t res_value_q = 0;

                        switch (op) {
                            case BIReductionOperation::ARG_IDX_MAX:
                            case BIReductionOperation::ARG_IDX_MIN:
                            case BIReductionOperation::MIN:
                            case BIReductionOperation::MAX: {
                                res_value = *(input_ptr + x);
                                break;
                            }
                            case BIReductionOperation::PROD: {
                                res_value = static_cast<T>(1.0f);
                                break;
                            }
                            default: {
                                res_value = static_cast<T>(0.0f);
                                break;
                            }
                        }
                        uint32_t res_idx = 0;

                        for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim) {
                            const T *in_ptr =
                                    reinterpret_cast<T *>(input.ptr() + x + in_info.strides_in_bytes()[axis] * dim);
                            switch (op) {
                                case BIReductionOperation::SUM: {
                                    res_value += *in_ptr;
                                    break;
                                }
                                case BIReductionOperation::MEAN_SUM: {
                                    res_value_q += *in_ptr;
                                    break;
                                }
                                case BIReductionOperation::SUM_SQUARE: {
                                    res_value += *in_ptr * *in_ptr;
                                    break;
                                }
                                case BIReductionOperation::PROD: {
                                    //de-quantize input
                                    if (std::is_same<T, uint8_t>::value) {
                                        res_value *= dequantize_qasymm8(*in_ptr, iq_info);
                                    } else {
                                        res_value *= dequantize_qasymm8_signed(*in_ptr, iq_info);
                                    }
                                    break;
                                }
                                case BIReductionOperation::ARG_IDX_MIN: {
                                    if (*in_ptr < res_value) {
                                        res_value = *in_ptr;
                                        res_idx = dim;
                                    }
                                    break;
                                }
                                case BIReductionOperation::ARG_IDX_MAX: {
                                    if (*in_ptr > res_value) {
                                        res_value = *in_ptr;
                                        res_idx = dim;
                                    }
                                    break;
                                }
                                case BIReductionOperation::MIN: {
                                    res_value = *in_ptr < res_value ? *in_ptr : res_value;
                                    break;
                                }
                                case BIReductionOperation::MAX: {
                                    res_value = *in_ptr > res_value ? *in_ptr : res_value;
                                    break;
                                }
                                default:
                                    BI_COMPUTE_ERROR("Not supported");
                            }
                        }

                        switch (op) {
                            case BIReductionOperation::MEAN_SUM: {
                                // Apply previously calculated coefficients (with rounding on aarch64)
#ifdef __aarch64__
                                const int32_t res = BatmanInfer::round(A * (static_cast<float>(res_value_q)) + B,
                                                                       BIRoundingPolicy::TO_NEAREST_EVEN);
#else  // defined(__aarch64__)
                            const int32_t res = A * (static_cast<float>(res_value_q)) + B;
#endif // __aarch64__
                                *reinterpret_cast<T *>(output.ptr() + x) = utils::cast::saturate_cast<T>(res);
                                break;
                            }
                            case BIReductionOperation::SUM: {
                                // Subtract accumulated offsets
                                res_value -= (in_info.dimension(axis) - 1) * iq_info.offset;
                                *reinterpret_cast<T *>(output.ptr() + x) = utils::cast::saturate_cast<T>(res_value);
                                break;
                            }
                            case BIReductionOperation::PROD: {
                                //re-quantize result
                                T res = 0;
                                if (std::is_same<T, uint8_t>::value) {
                                    res = quantize_qasymm8(res_value, iq_info);
                                } else {
                                    res = quantize_qasymm8_signed(res_value, iq_info);
                                }
                                *(reinterpret_cast<T *>(output.ptr() + x)) = res;
                                break;
                            }
                            case BIReductionOperation::ARG_IDX_MIN:
                            case BIReductionOperation::ARG_IDX_MAX: {
                                *(reinterpret_cast<uint32_t *>(output.ptr() + x * 4)) = res_idx;
                                break;
                            }
                            default:
                                *(reinterpret_cast<T *>(output.ptr() + x)) = res_value;
                        }
                    }
                },
                input, output);
        }
    };
}
