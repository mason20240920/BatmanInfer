//
// Created by Mason on 2025/1/17.
//
#include <cpu/kernels/softmax/generic/neon/qasymm8_signed.hpp>

#include <cpu/kernels/softmax/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        template<bool IS_LOG>
        void neon_qasymm8_signed_softmax(const BIITensor *in,
                                         void *const tmp,
                                         BIITensor *out,
                                         const float beta,
                                         int axis,
                                         const BIWindow &window,
                                         const void *lut_ptr) {
            BI_COMPUTE_UNUSED(lut_ptr);
            if (axis == 0) {
                return neon_softmax_x_quantized<qasymm8_signed_t, IS_LOG>(in, tmp, out, beta, axis, window);
            } else {
                return neon_softmax_non_x_quantized<qasymm8_signed_t, IS_LOG>(in, tmp, out, beta, axis, window);
            }
        }

        template void neon_qasymm8_signed_softmax<true>(const BIITensor *in,
                                                        void *const tmp,
                                                        BIITensor *out,
                                                        const float beta,
                                                        int axis,
                                                        const BIWindow &window,
                                                        const void *lut_ptr);

        template void neon_qasymm8_signed_softmax<false>(const BIITensor *in,
                                                         void *const tmp,
                                                         BIITensor *out,
                                                         const float beta,
                                                         int axis,
                                                         const BIWindow &window,
                                                         const void *lut_ptr);

    } // namespace cpu
} // namespace BatmanInfer