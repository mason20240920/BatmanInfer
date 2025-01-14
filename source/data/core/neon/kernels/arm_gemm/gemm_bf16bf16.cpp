//
// Created by Mason on 2025/1/14.
//

#include <data/core/neon/kernels/arm_gemm/bfloat.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp>


#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include <data/core/neon/kernels/arm_gemm/kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp>

#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

namespace BatmanGemm {

    static const GemmImplementation<bfloat16, bfloat16, bfloat16> gemm_bf16bf16_methods[] =
            {
#ifdef __aarch64__
#ifdef BI_COMPUTE_ENABLE_BF16
#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
                    GemmImplementation<bfloat16, bfloat16, bfloat16>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "a64_ffinterleaved_bf16fp32_mmla_8x12",
                        KernelWeightFormat::VL256_BL64,
                        [](const GemmArgs &args) { return args._ci->has_bf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16>(args); }
                    ),
#ifdef BI_COMPUTE_ENABLE_SVE
                    GemmImplementation<bfloat16, bfloat16, bfloat16>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
                        KernelWeightFormat::VL2VL_BL64,
                        [](const GemmArgs &args) { return args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16>(args); }
                    ),
#endif // BI_COMPUTE_ENABLE_SVE
#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // BI_COMPUTE_ENABLE_BF16
#endif // __aarch64__
                    {
                            GemmMethod::DEFAULT,
                            "",
                            nullptr,
                            nullptr,
                            nullptr
                    }
            };

    template<>
    const GemmImplementation<bfloat16, bfloat16, bfloat16> *gemm_implementation_list<bfloat16, bfloat16, bfloat16>() {
        return gemm_bf16bf16_methods;
    }

/* Explicitly instantiate the external functions for these types. */
    template UniqueGemmCommon<bfloat16, bfloat16, bfloat16>
    gemm<bfloat16, bfloat16, bfloat16, Nothing>(const GemmArgs &args, const Nothing &);

    template bool has_opt_gemm<bfloat16, bfloat16, bfloat16, Nothing>(WeightFormat &weight_format, const GemmArgs &args,
                                                                      const Nothing &);

    template KernelDescription
    get_gemm_method<bfloat16, bfloat16, bfloat16, Nothing>(const GemmArgs &args, const Nothing &);

    template std::vector<KernelDescription>
    get_compatible_kernels<bfloat16, bfloat16, bfloat16, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm