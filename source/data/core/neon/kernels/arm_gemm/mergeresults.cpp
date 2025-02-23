//
// Created by Mason on 2025/1/14.
//

#include <algorithm>

#include <neon/neon_defines.h>

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <data/core/neon/kernels/arm_gemm/asmlib.hpp>
#include <data/core/neon/kernels/arm_gemm/bfloat.hpp>
#include <data/core/neon/kernels/arm_gemm/utils.hpp>

namespace BatmanGemm {
    template<unsigned int twidth, unsigned int height, bool sve = false, typename Tin, typename Tout>
    void MergeResults(Tout *out, const Tin *in, int ldc, int y0, int ymax, int x0, int xmax, const Tout *bias,
                      Activation act, bool append) {
        // NOTE: The following code is disabled to avoid calling get_vector_length(), so templated MergeResults will not
        // be correct for SVE cases.  This is OK as we have specialisations for all needed SVE cases anyway.
        //
        // For SVE cases, multiply the width up by the vector length.
        // Use the *input* type to determine this, since this will be what the kernel operated on.
        // const int width = twidth * (sve ? get_vector_length<Tin>() : 1);
        const int width = twidth;

        const int full_y_blocks = (ymax - y0) / height;
        const int y_remainder = (ymax - y0) % height;
        const int y_blocks = full_y_blocks + (y_remainder ? 1 : 0);

        const int full_x_blocks = (xmax - x0) / width;
        const int x_remainder = (xmax - x0) % width;
        const int x_blocks = full_x_blocks + (x_remainder ? 1 : 0);

        for (int y_block = 0; y_block < y_blocks; y_block++) {
            int ybase = y0 + (y_block * height);

            int fill_rows = (y_block < full_y_blocks) ? height : y_remainder;

            for (int x_block = 0; x_block < x_blocks; x_block++) {
                int xbase = x0 + (x_block * width);

                int fill_cols = (x_block < full_x_blocks) ? width : x_remainder;

                for (int row = 0; row < fill_rows; row++) {
                    for (int col = 0; col < fill_cols; col++) {
                        Tout &r = out[(ybase + row) * ldc + xbase + col];
                        Tout v = in[row * width + col];

                        if (append) {
                            v += r;
                        }

                        if (bias) {
                            v += bias[xbase + col];
                        }

                        switch (act.type) {
                            default:
                            case Activation::Type::None:
                                break;

                            case Activation::Type::ReLU:
                                v = std::max(v, static_cast<Tout>(0));
                                break;

                            case Activation::Type::BoundedReLU:
                                v = std::max(std::min(v, static_cast<Tout>(act.param1)), static_cast<Tout>(0));
                                break;
                        }

                        r = v;
                    }
                }

                in += (width * height);
            }
        }
    }

#include <data/core/neon/kernels/arm_gemm/merges/list.hpp>

/* Cortex-A53 8x6 SGEMM kernel uses a templated merge as the optimized merge
 * generator cannot cope with the width (6) not being a multiple of VL (4). */
#ifdef __aarch64__

    template void
    MergeResults<6u, 8u, false, float, float>(float *, float const *, int, int, int, int, int, float const *,
                                              Activation, bool);

#endif

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS)

    template void
    MergeResults<12u, 8u, false, float, __fp16>(__fp16 *, float const *, int, int, int, int, int, __fp16 const *,
                                                Activation, bool);

#endif

#if defined(__arm__) && defined(__ARM_FP16_ARGS)
    template void MergeResults<8u, 6u, false, float, __fp16>(__fp16*, float const*, int, int, int, int, int, __fp16 const*, Activation, bool);
#endif

#if defined(__arm__) && defined(ARM_COMPUTE_ENABLE_BF16)
    template void MergeResults<8u, 6u, false, float, bfloat16>(bfloat16*, float const*, int, int, int, int, int, bfloat16 const*, Activation, bool);
#endif
}