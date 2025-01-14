//
// Created by Mason on 2025/1/14.
//

#ifdef __aarch64__

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <cpu/kernels/assembly/bi_gemm_common.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp>

#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s16_8x12.hpp"

namespace BatmanGemm {

    static const GemmImplementation<int16_t, int16_t, int32_t> gemm_s16_methods[] = {
            {
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_gemm_s16_8x12",
                    nullptr,
                    nullptr,
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_gemm_s16_8x12, int16_t, int16_t, int32_t>(args);
                    }
            },
            {
                    GemmMethod::DEFAULT,
                    "",
                    nullptr,
                    nullptr,
                    nullptr
            }
    };

    template<>
    const GemmImplementation<int16_t, int16_t, int32_t> *gemm_implementation_list<int16_t, int16_t, int32_t>() {
        return gemm_s16_methods;
    }

/* Explicitly instantiate the external functions for these types. */
    template UniqueGemmCommon<int16_t, int16_t, int32_t>
    gemm<int16_t, int16_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);

    template bool has_opt_gemm<int16_t, int16_t, int32_t, Nothing>(WeightFormat &weight_format, const GemmArgs &args,
                                                                   const Nothing &);

    template KernelDescription
    get_gemm_method<int16_t, int16_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);

    template std::vector<KernelDescription>
    get_compatible_kernels<int16_t, int16_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm

#endif // __aarch64__