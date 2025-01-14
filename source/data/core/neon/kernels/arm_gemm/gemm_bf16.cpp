//
// Created by Mason on 2025/1/14.
//

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <data/core/neon/kernels/arm_gemm/bfloat.hpp>
#include <cpu/kernels/assembly/bi_gemm_common.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid_indirect.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_batched.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_pretransposed.hpp>

#include <data/core/neon/kernels/arm_gemm/kernels/a32_sgemm_8x6.hpp>

#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include <data/core/neon/kernels/arm_gemm/kernels/a64_ffhybrid_bf16fp32_mmla_6x16.hpp>
#include <data/core/neon/kernels/arm_gemm/kernels/a64_ffinterleaved_bf16fp32_dot_8x12.hpp>
#include <data/core/neon/kernels/arm_gemm/kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp>

#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include <data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_bf16fp32_dot_6x16.hpp>
#include <data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_bf16fp32_mmla_6x16.hpp>
#include <data/core/neon/kernels/arm_gemm/kernels/a64_interleaved_bf16fp32_dot_8x12.hpp>
#include <data/core/neon/kernels/arm_gemm/kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp>
#include <data/core/neon/kernels/arm_gemm/kernels/a64_sgemm_8x12.hpp>

namespace BatmanGemm {

    static const GemmImplementation<bfloat16, bfloat16, float> gemm_bf16_methods[] =
            {
#ifdef BI_COMPUTE_ENABLE_BF16
#ifdef __aarch64__
#ifdef BI_COMPUTE_ENABLE_SVE
#ifdef BI_COMPUTE_ENABLE_SME2
                    // SME kernels
                    {
                        GemmMethod::GEMM_HYBRID,
                        "sme2_gemv_bf16fp32_dot_16VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input; },
                        nullptr,
                        [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_bf16fp32_dot_16VL, bfloat16, float>(args); }
                    },
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2(); },
                        [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                                   return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL, bfloat16, float>(args); }
                    },
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2(); },
                        [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                                   return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL, bfloat16, float>(args); }
                    },
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2(); },
                        nullptr,
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL, bfloat16, float>(args); }
                    },
#endif // ARM_COMPUTE_ENABLE_SME2
                    // gemm_bf16_interleaved
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_bf16fp32_mmla_8x3VL",
                        [](const GemmArgs &args) { return args._ci->has_svebf16() && (args._Ksize>4); },
                        [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_hybrid_bf16fp32_mmla_6x4VL",
                        [](const GemmArgs &args) { return args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_bf16fp32_mmla_6x4VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_mmla_6x4VL, bfloat16, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_hybrid_bf16fp32_dot_6x4VL",
                        [](const GemmArgs &args) { return args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_bf16fp32_dot_8x3VL",
                        [](const GemmArgs &args) { return args._ci->has_svebf16() && (args._Ksize>2); },
                        [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, bfloat16, float>(args); }
                    ),
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
                        KernelWeightFormat::VL2VL_BL64,
                        [](const GemmArgs &args) { return args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_ffinterleaved_bf16fp32_dot_8x3VL",
                        KernelWeightFormat::VL1VL_BL32,
                        [](const GemmArgs &args) { return args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_dot_8x3VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_dot_8x3VL, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_ffhybrid_bf16fp32_mmla_6x4VL",
                        KernelWeightFormat::VL2VL_BL64,
                        [](const GemmArgs &args) { return args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_bf16fp32_mmla_6x4VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_bf16fp32_mmla_6x4VL, bfloat16, float>(args); }
                    ),
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_SVE
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_bf16fp32_mmla_6x16",
                            [](const GemmArgs &args) { return args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmHybridIndirect<cls_a64_hybrid_bf16fp32_mmla_6x16, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_mmla_6x16, bfloat16, bfloat16, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_interleaved_bf16fp32_mmla_8x12",
                            [](const GemmArgs &args) { return args._ci->has_bf16() && (args._Ksize > 4); },
                            [](const GemmArgs &args) {
                                return GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_bf16fp32_dot_6x16",
                            [](const GemmArgs &args) { return args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, bfloat16, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_interleaved_bf16fp32_dot_8x12",
                            [](const GemmArgs &args) { return args._ci->has_bf16() && (args._Ksize > 2); },
                            [](const GemmArgs &args) {
                                return GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, bfloat16, float>(
                                        args);
                            }
                    ),
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "a64_ffinterleaved_bf16fp32_mmla_8x12",
                        KernelWeightFormat::VL256_BL64,
                        [](const GemmArgs &args) { return args._ci->has_bf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "a64_ffhybrid_bf16fp32_mmla_6x16",
                        KernelWeightFormat::VL256_BL64,
                        [](const GemmArgs &args) { return args._ci->has_bf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_bf16fp32_mmla_6x16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_bf16fp32_mmla_6x16, bfloat16, float>(args); }
                    ),
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "a64_ffinterleaved_bf16fp32_dot_8x12",
                        KernelWeightFormat::VL128_BL32,
                        [](const GemmArgs &args) { return args._ci->has_bf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_dot_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_dot_8x12, bfloat16, float>(args); }
                    ),
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
                    GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_sgemm_8x12",
                            nullptr,
                            [](const GemmArgs &args) {
                                return GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, bfloat16, float>(args);
                            }
                    ),
#elif defined(__arm__)
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sgemm_8x6",
                        nullptr,
                        nullptr,
                        [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, bfloat16, bfloat16, float>(args); }
                    },
#else
# error "Unknown Architecture"
#endif
#endif // BI_COMPUTE_ENABLE_BF16
                    {
                            GemmMethod::DEFAULT,
                            "",
                            nullptr,
                            nullptr,
                            nullptr
                    }
            };

    template<>
    const GemmImplementation<bfloat16, bfloat16, float> *gemm_implementation_list<bfloat16, bfloat16, float>() {
        return gemm_bf16_methods;
    }

    /* Explicitly instantiate the external functions for these types. */
    template UniqueGemmCommon<bfloat16, bfloat16, float>
    gemm<bfloat16, bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);

    template bool has_opt_gemm<bfloat16, bfloat16, float, Nothing>(WeightFormat &weight_format, const GemmArgs &args,
                                                                   const Nothing &);

    template KernelDescription
    get_gemm_method<bfloat16, bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);

    template std::vector<KernelDescription>
    get_compatible_kernels<bfloat16, bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace BatmanGemm