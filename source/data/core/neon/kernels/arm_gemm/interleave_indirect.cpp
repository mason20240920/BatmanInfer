//
// Created by Mason on 2025/1/14.
//

#include "data/core/neon/kernels/arm_gemm/asmlib.hpp"
#include <cpu/kernels/assembly/bi_convolution_parameters.hpp>
#include "data/core/neon/kernels/arm_gemm/convolver.hpp"
#include "data/core/neon/kernels/arm_gemm/interleave_indirect.hpp"
#include "data/core/neon/kernels/arm_gemm/bfloat.hpp"

#if !defined(_WIN64) && !defined(__OpenBSD__)

#include <alloca.h>

#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include <neon/neon_defines.h>

#include "data/core/neon/kernels/arm_gemm/utils.hpp"

namespace BatmanGemm {

#include "data/core/neon/kernels/arm_gemm/interleave_indirect_impl.hpp"

#include "data/core/neon/kernels/arm_gemm/indirect-interleaves/list.hpp"

/**** Instantiate needed implementations ****/

/* AArch32 */
#ifdef __arm__
    /* FP32 */
/* Arm® Neon™ implementation (height 6) */
template void IndirectInterleave<6, 1, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<6, 1, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<6, 1, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* FP16 */
#if __ARM_FP16_ARGS
/* Arm® Neon™ implementation using FP32 kernel (height 6) */
template void IndirectInterleave<6, 1, VLType::None>(float *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<6, 1, VLType::None>(float *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<6, 1, VLType::None>(float *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
#endif /* __ARM_FP16_ARGS */

/* BF16 */
/* Arm® Neon™ implementation using FP32 kernel */
template void IndirectInterleave<6, 1, VLType::None>(float *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<6, 1, VLType::None>(float *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<6, 1, VLType::None>(float *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
#endif

/* AArch64 */
#ifdef __aarch64__

/* FP32 */
/* Arm® Neon™/SVE implementation (height 8) */
    template void
    IndirectInterleave<8, 1, VLType::None>(float *, const float *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int,
                                              unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

#if defined(BI_COMPUTE_ENABLE_SVE) && defined(BI_COMPUTE_ENABLE_SVEF32MM)
    /* FMMLA */
template void IndirectInterleave<8, 2, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 2, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 2, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
#endif // BI_COMPUTE_ENABLE_SVE && BI_COMPUTE_ENABLE_SVEF32MM

/* FP16 */
#if defined(FP16_KERNELS) || defined(BI_COMPUTE_ENABLE_FP16)

    template void
    IndirectInterleave<8, 1, VLType::None>(__fp16 *, const __fp16 *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int,
                                              unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

#endif // FP16_KERNELS ar BI_COMPUTE_ENABLE_FP16

    template void
    IndirectInterleave<8, 1, VLType::None>(float *, const __fp16 *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(float *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int,
                                              unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(float *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* BF16 */
/* Arm® Neon™/SVE BFDOT */
#ifdef BI_COMPUTE_ENABLE_BF16

    template void
    IndirectInterleave<8, 2, VLType::None>(bfloat16 *, const bfloat16 *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 2, VLType::None>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 2, VLType::None>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

    template void
    IndirectInterleave<8, 4, VLType::None>(bfloat16 *, const bfloat16 *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 4, VLType::None>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 4, VLType::None>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

    template void
    IndirectInterleave<8, 4, VLType::None>(bfloat16 *, const float *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 4, VLType::None>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int,
                                              unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    Interleave<8, 4, VLType::None>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

#endif // BI_COMPUTE_ENABLE_BF16

/* Arm® Neon™/SVE using FP32 kernel */
    template void
    IndirectInterleave<8, 1, VLType::None>(float *, const bfloat16 *const *const *, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(float *, const bfloat16 *, size_t, const convolver<bfloat16> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(float *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* INT16 */
    template void
    IndirectInterleave<8, 1, VLType::None>(int16_t *, const int16_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(int16_t *, const int16_t *, size_t, const convolver<int16_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(int16_t *, const int16_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

    template void
    IndirectInterleave<8, 1, VLType::None>(uint16_t *, const uint16_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(uint16_t *, const uint16_t *, size_t, const convolver<uint16_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(uint16_t *, const uint16_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* INT8 */
/* Arm® Neon™ SMLA/SMLAL (height 4, block 16) */
    template void
    IndirectInterleave<4, 16, VLType::None>(int8_t *, const int8_t *const *const *, unsigned int, unsigned int,
                                            unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<4, 16, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &,
                                               unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                               bool, int32_t);

    template void
    Interleave<4, 16, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                    unsigned int, bool, int32_t);

/* Arm® Neon™ SDOT (height 8, block 4) */
    template void
    IndirectInterleave<8, 4, VLType::None>(int8_t *, const int8_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 4, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 4, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* MMLA SMMLA (height 8, block 8) */
    template void
    IndirectInterleave<8, 8, VLType::None>(int8_t *, const int8_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 8, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 8, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* Arm® Neon™ SDOT (height 8, block 1) */
    template void
    IndirectInterleave<8, 1, VLType::None>(int16_t *, const int8_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(int16_t *, const int8_t *, size_t, const convolver<int8_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(int16_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* Arm® Neon™ SMLA/SMLAL (height 4, block 16) */
    template void
    IndirectInterleave<4, 16, VLType::None>(uint8_t *, const uint8_t *const *const *, unsigned int, unsigned int,
                                            unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

    template void
    ConvolutionInterleave<4, 16, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &,
                                               unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                               bool, int32_t);

    template void
    Interleave<4, 16, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                    unsigned int, bool, int32_t);


/* Arm® Neon™ SDOT (height 8, block 4) */
    template void
    IndirectInterleave<8, 4, VLType::None>(uint8_t *, const uint8_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 4, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 4, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* MMLA SMMLA (height 8, block 8) */
    template void
    IndirectInterleave<8, 8, VLType::None>(uint8_t *, const uint8_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 8, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 8, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

/* Arm® Neon™ 16-bit (height 8, block 1) */
    template void
    IndirectInterleave<8, 1, VLType::None>(uint16_t *, const uint8_t *const *const *, unsigned int, unsigned int,
                                           unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool,
                                           int32_t);

    template void
    ConvolutionInterleave<8, 1, VLType::None>(uint16_t *, const uint8_t *, size_t, const convolver<uint8_t> &,
                                              unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                              bool, int32_t);

    template void
    Interleave<8, 1, VLType::None>(uint16_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int,
                                   unsigned int, bool, int32_t);

#endif // __aarch64__

} // namespace arm_gemm