//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <data/core/bi_helpers.hpp>

#include <data/core/neon/bi_neon_math.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>
#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
#ifdef __aarch64__
        namespace {
// These helper functions are added because vaddv does not exist for fp16,
// and, therefore, is not part of the wrapper::vaddv interface.
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

            inline float16_t wrapper_vaddv(const float16x8_t &a, int sum_stages) {
                auto sum_res = wrapper::vpadd(wrapper::vgethigh(a), wrapper::vgetlow(a));
                for (int i = 0; i < sum_stages; ++i) {
                    sum_res = wrapper::vpadd(sum_res, sum_res);
                }
                return wrapper::vgetlane(sum_res, 0);
            }

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

            inline float wrapper_vaddv(const float32x4_t &a, int sum_stages) {
                BI_COMPUTE_UNUSED(sum_stages);
                return wrapper::vaddv(a);
            }
        } // namespace
#endif // __aarch64__


        // The template implementation for float data types is stored in the header file because
        // we need all fp16 instantiated code to live in fp16.cpp files.
        template<typename T, bool IS_LOG>
        void neon_softmax_x_float(const BIITensor *in, void *const tmp, BIITensor *out, float beta, int axis,
                                  const BIWindow &window) {
            BI_COMPUTE_UNUSED(axis);
            BI_COMPUTE_UNUSED(tmp);

            const int input_width = in->info()->valid_region().shape.x();

            BIIterator in_it(in, window);
            BIIterator out_it(out, window);

            /** SIMD vector tag type. */
            using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

            constexpr int vec_size = 16 / sizeof(T);

            const int sum_stages = log2(vec_size >> 1);

            const auto beta_vec = wrapper::vdup_n(static_cast<T>(beta), ExactTagType{});

            execute_window_loop(
                    window,
                    [&](const BICoordinates &) {
                        /* Get pointers */
                        const T *in_ptr = reinterpret_cast<const T *>(in_it.ptr());
                        T *out_ptr = reinterpret_cast<T *>(out_it.ptr());

                        T max_val;

                        /* Compute Max */
                        {
                            // Init max value
                            auto vec_max = wrapper::vdup_n(support::cpp11::lowest<T>(), ExactTagType{});
                            int x = 0;

                            for (; x <= (input_width - vec_size); x += vec_size) {
                                const auto current_value = wrapper::vloadq(in_ptr + x);
                                vec_max = wrapper::vmax(vec_max, current_value);
                            }

#ifdef __aarch64__
                            max_val = wrapper::vmaxv(vec_max);
#else  // __aarch64__
                            auto carry_max = wrapper::vpmax(wrapper::vgethigh(vec_max), wrapper::vgetlow(vec_max));

                for (int i = 0; i < sum_stages; ++i)
                {
                    carry_max = wrapper::vpmax(carry_max, carry_max);
                }

                max_val      = wrapper::vgetlane(carry_max, 0);
#endif // __aarch64__

                            // Compute left-over elements
                            for (; x < input_width; ++x) {
                                max_val = std::max(*(in_ptr + x), max_val);
                            }
                        } // compute max

                        T sum_transformed{};

                        /* Compute exponentials and sum */
                        {
                            /* Get max value */
                            const auto vec_max = wrapper::vdup_n(max_val, ExactTagType{});

                            /* Init sum to zero */
                            auto vec_sum = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});

                            /* Loop over row and compute exponentials and sum */
                            int x = 0;
                            for (; x <= (input_width - vec_size); x += vec_size) {
                                auto vec_elements = wrapper::vloadq(in_ptr + x);
                                vec_elements = wrapper::vsub(vec_elements, vec_max);
                                if (IS_LOG) {
                                    vec_elements = wrapper::vmul(vec_elements, beta_vec);
                                    vec_sum = wrapper::vadd(vec_sum, wrapper::vexpq(vec_elements));
                                } else {
                                    vec_elements = wrapper::vexpq(wrapper::vmul(vec_elements, beta_vec));
                                    vec_sum = wrapper::vadd(vec_sum, vec_elements);
                                }
                                wrapper::vstore(out_ptr + x, vec_elements);
                            }

                            /* Reduce sum */
                            T sum{};
#ifdef __aarch64__
                            sum = wrapper_vaddv(vec_sum, sum_stages);
#else  // __aarch64__
                            auto sum_res = wrapper::vpadd(wrapper::vgethigh(vec_sum), wrapper::vgetlow(vec_sum));
                for (int i = 0; i < sum_stages; ++i)
                {
                    sum_res = wrapper::vpadd(sum_res, sum_res);
                }
                sum = wrapper::vgetlane(sum_res, 0);
#endif // __aarch64__

                            /* Run remaining elements */
                            for (; x < input_width; ++x) {
                                T element{};

                                if (IS_LOG) {
                                    element = (in_ptr[x] - max_val) * beta;
                                    sum += std::exp(element);
                                } else {
                                    element = std::exp((in_ptr[x] - max_val) * beta);
                                    sum += element;
                                }

                                out_ptr[x] = element;
                            }

                            if (!IS_LOG) {
                                sum_transformed = T(1) / sum;
                            } else {
                                sum_transformed = static_cast<T>(std::log(sum));
                            }
                        } // Compute exponentials and sum

                        /* Normalize exponentials */
                        {
                            const auto sum_vec = wrapper::vdup_n(static_cast<T>(sum_transformed), ExactTagType{});

                            /* Loop over row and compute softmax */
                            int x = 0;
                            for (; x <= (input_width - vec_size); x += vec_size) {
                                const auto vec_in = wrapper::vloadq(out_ptr + x);
                                if (IS_LOG) {
                                    wrapper::vstore(out_ptr + x, wrapper::vsub(vec_in, sum_vec));
                                } else {
                                    wrapper::vstore(out_ptr + x, wrapper::vmul(vec_in, sum_vec));
                                }
                            }

                            /* Run remaining elements */
                            for (; x < input_width; ++x) {
                                if (IS_LOG) {
                                    out_ptr[x] = out_ptr[x] - sum_transformed;
                                } else {
                                    out_ptr[x] = out_ptr[x] * sum_transformed;
                                }
                            }
                        } // Normalize exponentials
                    },
                    in_it, out_it);
        }

        template<typename T, bool IS_LOG>
        void neon_softmax_non_x_float(
                const BIITensor *in, void *const tmp, BIITensor *out, float beta, int axis, const BIWindow &window) {
            BI_COMPUTE_UNUSED(tmp);

            BIIterator in_it(in, window);
            BIIterator out_it(out, window);

            /** SIMD vector tag type. */
            using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

            const auto beta_vec = wrapper::vdup_n(static_cast<T>(beta), ExactTagType{});
            constexpr int vec_size = 16 / sizeof(T);
            const BIITensorInfo *in_info = in->info();
            const BIITensorInfo *out_info = out->info();
            const int x_width = in_info->valid_region().shape.x();
            const unsigned int in_axis_stride = in_info->strides_in_bytes()[axis];
            const unsigned int out_axis_stride = out_info->strides_in_bytes()[axis];
            const int axis_width = in_info->dimension(axis);

            execute_window_loop(
                    window,
                    [&](const BICoordinates &winCoords) {
                        const bool vector_exceeds_bounds = (winCoords[0] + vec_size) > x_width;

                        /* Get pointers */
                        const uint8_t *in_ptr = in_it.ptr();
                        uint8_t *out_ptr = out_it.ptr();

                        // Init max value
                        auto vec_max = wrapper::vdup_n(support::cpp11::lowest<T>(), ExactTagType{});

                        /* Compute Max */
                        {
                            if (!vector_exceeds_bounds) {
                                int i = 0;
                                for (; i < axis_width; ++i) {
                                    const auto current_value =
                                            wrapper::vloadq(reinterpret_cast<const T *>((i * in_axis_stride) + in_ptr));
                                    vec_max = wrapper::vmax(vec_max, current_value);
                                }
                            } else {
                                int i = 0;
                                for (; i < axis_width; ++i) {
                                    const T *const base_ptr_in = reinterpret_cast<const T *>((i * in_axis_stride) +
                                                                                             in_ptr);
                                    int j = 0;
                                    for (; j < (x_width - winCoords[0]); ++j) {
                                        const auto current_value = *(base_ptr_in + j);
                                        vec_max[j] = std::max(vec_max[j], current_value);
                                    }
                                }
                            }
                        } // compute max

                        auto vec_sum_transformed = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});

                        auto vec_elements = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});
                        /* Init sum to zero */
                        auto vec_sum = wrapper::vdup_n(static_cast<T>(0), ExactTagType{});

                        /* Compute exponentials and sum */
                        {
                            if (!vector_exceeds_bounds) {
                                const auto vec_one = wrapper::vdup_n(static_cast<T>(1), ExactTagType{});
                                /* Loop over row and compute exponentials and sum */
                                int i = 0;
                                for (; i < axis_width; ++i) {
                                    vec_elements = wrapper::vloadq(
                                            reinterpret_cast<const T *>((i * in_axis_stride) + in_ptr));
                                    vec_elements = wrapper::vsub(vec_elements, vec_max);
                                    if (IS_LOG) {
                                        vec_elements = wrapper::vmul(vec_elements, beta_vec);
                                        vec_sum = wrapper::vadd(vec_sum, wrapper::vexpq(vec_elements));
                                    } else {
                                        vec_elements = wrapper::vexpq(wrapper::vmul(vec_elements, beta_vec));
                                        vec_sum = wrapper::vadd(vec_sum, vec_elements);
                                    }

                                    wrapper::vstore(reinterpret_cast<T *>((i * out_axis_stride) + out_ptr),
                                                    vec_elements);
                                }

                                if (!IS_LOG) {
                                    vec_sum_transformed = wrapper::vdiv(vec_one, vec_sum);
                                } else {
                                    vec_sum_transformed = wrapper::vlog(vec_sum);
                                }
                            } else {
                                int i = 0;
                                for (; i < axis_width; ++i) {
                                    const T *const base_ptr_in = reinterpret_cast<const T *>((i * in_axis_stride) +
                                                                                             in_ptr);
                                    T *const base_ptr_out = reinterpret_cast<T *>((i * out_axis_stride) + out_ptr);
                                    int j = 0;
                                    for (; j < (x_width - winCoords[0]); ++j) {
                                        vec_elements[j] = *(base_ptr_in + j);
                                        vec_elements[j] -= vec_max[j];
                                        if (IS_LOG) {
                                            vec_elements[j] *= beta;
                                            vec_sum[j] += std::exp(vec_elements[j]);
                                        } else {
                                            vec_elements[j] = std::exp(vec_elements[j] * beta);
                                            vec_sum[j] += vec_elements[j];
                                        }
                                        *(base_ptr_out + j) = vec_elements[j];
                                    }
                                }
                                int j = 0;
                                for (; j < (x_width - winCoords[0]); ++j) {
                                    if (!IS_LOG) {
                                        vec_sum_transformed[j] = 1 / vec_sum[j];
                                    } else {
                                        vec_sum_transformed[j] = std::log(vec_sum[j]);
                                    }
                                }
                            }
                        } // Compute exponentials and sum

                        /* Normalize exponentials */
                        {
                            if (!vector_exceeds_bounds) {
                                /* Loop over row and compute softmax */
                                int i = 0;
                                for (; i < axis_width; ++i) {
                                    T *const base_ptr_out = reinterpret_cast<T *>((i * out_axis_stride) + out_ptr);
                                    auto vec_in = wrapper::vloadq(base_ptr_out);
                                    if (IS_LOG) {
                                        wrapper::vstore(base_ptr_out, wrapper::vsub(vec_in, vec_sum_transformed));
                                    } else {
                                        wrapper::vstore(base_ptr_out, wrapper::vmul(vec_in, vec_sum_transformed));
                                    }
                                }
                            } else {
                                int i = 0;
                                for (; i < axis_width; ++i) {
                                    T *const base_ptr_out = reinterpret_cast<T *>((i * out_axis_stride) + out_ptr);
                                    int j = 0;
                                    for (; j < (x_width - winCoords[0]); ++j) {
                                        if (IS_LOG) {
                                            *(base_ptr_out + j) -= vec_sum_transformed[j];
                                        } else {
                                            *(base_ptr_out + j) *= vec_sum_transformed[j];
                                        }
                                    }
                                }
                            }
                        } // Normalize exponentials
                    },
                    in_it, out_it);
        }

        template<typename T, bool IS_LOG>
        void neon_softmax_x_quantized(
                const BIITensor *in, void *const tmp, BIITensor *out, float beta, int axis, const BIWindow &window);

        template<typename T, bool IS_LOG>
        void neon_softmax_non_x_quantized(
                const BIITensor *in, void *const tmp, BIITensor *out, float beta, int axis, const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer