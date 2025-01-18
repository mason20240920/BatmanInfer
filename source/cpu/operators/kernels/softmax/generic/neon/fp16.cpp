//
// Created by Mason on 2025/1/17.
//

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <cpu/kernels/softmax/generic/neon/fp16.hpp>

#include <cpu/bi_cpu_types.hpp>
#include <cpu/kernels/softmax/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        template<bool IS_LOG>
        void neon_fp16_softmax(const BIITensor *in,
                               void *const tmp,
                               BIITensor *out,
                               const float beta,
                               int axis,
                               const BIWindow &window,
                               const void *lut_ptr) {
            BI_COMPUTE_UNUSED(lut_ptr);
            if (axis == 0) {
                return neon_softmax_x_float<float16_t, IS_LOG>(in, tmp, out, beta, axis, window);
            } else {
                return neon_softmax_non_x_float<float16_t, IS_LOG>(in, tmp, out, beta, axis, window);
            }
        }

        template void neon_fp16_softmax<true>(const BIITensor *in,
                                              void *const tmp,
                                              BIITensor *out,
                                              const float beta,
                                              int axis,
                                              const BIWindow &window,
                                              const void *lut_ptr);

        template void neon_fp16_softmax<false>(const BIITensor *in,
                                               void *const tmp,
                                               BIITensor *out,
                                               const float beta,
                                               int axis,
                                               const BIWindow &window,
                                               const void *lut_ptr);
    }
}

#endif