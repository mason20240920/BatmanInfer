//
// Created by Mason on 2025/1/14.
//

// This can only be built if the target/compiler supports FP16 arguments.
#if defined(__aarch64__) && (defined(ENABLE_FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

#include <cpu/kernels/assembly/bi_gemm_common.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid_indirect.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_pretransposed.hpp>

#include <data/core/neon/kernels/arm_gemm/kernels/a32_sgemm_8x6.hpp>

#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffhybrid_fp16_mla_6x32.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffinterleaved_fp16_mla_8x24.hpp"

#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include "data/core/neon/kernels/arm_gemm/kernels/a64_hgemm_8x24.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_fp16_mla_6x32.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_sgemm_8x12.hpp"

namespace BatmanGemm {

    static const GemmImplementation<__fp16, __fp16, __fp16> gemm_fp16_methods[] = {
#ifdef BI_COMPUTE_ENABLE_SVE
#ifdef BI_COMPUTE_ENABLE_SME2
            {
                GemmMethod::GEMM_HYBRID,
                "sme2_gemv_fp16fp32fp16_dot_16VL",
                [](const GemmArgs &args) { return args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input; },
                nullptr,
                [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp16fp32fp16_dot_16VL, __fp16, __fp16>(args); }
            },
            {
                GemmMethod::GEMM_INTERLEAVED,
                "sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL",
                [](const GemmArgs &args) { return args._ci->has_sme2(); },
                [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                           return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
                [](const GemmArgs &args) { return new GemmInterleaved<cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL, __fp16, __fp16, __fp16, Nothing, false, false, false, true>(args); }
            },
            {
                GemmMethod::GEMM_INTERLEAVED,
                "sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL",
                [](const GemmArgs &args) { return args._ci->has_sme2(); },
                [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                           return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
                [](const GemmArgs &args) { return new GemmInterleaved<cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL, __fp16, __fp16, __fp16, Nothing, false, false, false, true>(args); }
            },
            {
                GemmMethod::GEMM_INTERLEAVED,
                "sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL",
                [](const GemmArgs &args) { return args._ci->has_sme2(); },
                nullptr,
                [](const GemmArgs &args) { return new GemmInterleaved<cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL, __fp16, __fp16, __fp16, Nothing, false, false, false, true>(args); }
            },
#endif // ARM_COMPUTE_ENABLE_SME2
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                GemmMethod::GEMM_HYBRID,
                "sve_hybrid_fp16_mla_6x4VL",
                [](const GemmArgs &args) { return args._ci->has_sve(); },
                [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp16_mla_6x4VL, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
                [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp16_mla_6x4VL, __fp16, __fp16, __fp16>(args); }
            ),
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                GemmMethod::GEMM_INTERLEAVED,
                "sve_interleaved_fp16_mla_8x3VL",
                [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize > 4); },
                [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_fp16_mla_8x3VL, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
                [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp16_mla_8x3VL, __fp16, __fp16, __fp16>(args); }
            ),
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                GemmMethod::GEMM_INTERLEAVED,
                "sve_ffinterleaved_fp16_mla_8x3VL",
                KernelWeightFormat::VL1VL_BL16,
                [](const GemmArgs &args) { return args._ci->has_sve(); },
                [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp16_mla_8x3VL, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
                [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp16_mla_8x3VL, __fp16, __fp16>(args); }
            ),
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                GemmMethod::GEMM_HYBRID,
                "sve_ffhybrid_fp16_mla_6x4VL",
                KernelWeightFormat::VL1VL_BL16,
                [](const GemmArgs &args) { return args._ci->has_sve(); },
                [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16_mla_6x4VL, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
                [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16_mla_6x4VL, __fp16, __fp16>(args); }
            ),
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_SVE
#if defined(__aarch64__)
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                    GemmMethod::GEMM_HYBRID,
                    "a64_hybrid_fp16_mla_6x32",
                    [](const GemmArgs &args) { return args._ci->has_fp16(); },
                    [](const GemmArgs &args) {
                        return GemmHybridIndirect<cls_a64_hybrid_fp16_mla_6x32, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmHybridIndirect<cls_a64_hybrid_fp16_mla_6x32, __fp16, __fp16, __fp16>(args);
                    }
            ),
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_hgemm_8x24",
                    [](const GemmArgs &args) { return args._ci->has_fp16(); },
                    [](const GemmArgs &args) {
                        return GemmInterleaved<cls_a64_hgemm_8x24, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_hgemm_8x24, __fp16, __fp16, __fp16>(args);
                    }
            ),
#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_ffinterleaved_fp16_mla_8x24",
                    KernelWeightFormat::VL128_BL16,
                    [](const GemmArgs &args) { return args._ci->has_fp16(); },
                    [](const GemmArgs &args) {
                        return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp16_mla_8x24, __fp16, __fp16>::estimate_cycles<__fp16>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp16_mla_8x24, __fp16, __fp16>(
                                args);
                    }
            ),
            GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
                    GemmMethod::GEMM_HYBRID,
                    "a64_ffhybrid_fp16_mla_6x32",
                    KernelWeightFormat::VL128_BL16,
                    [](const GemmArgs &args) { return args._ci->has_fp16(); },
                    [](const GemmArgs &args) {
                        return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16_mla_6x32, __fp16, __fp16>::estimate_cycles<__fp16>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16_mla_6x32, __fp16, __fp16>(args);
                    }
            ),
#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
            {
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_sgemm_8x12",
                    nullptr,
                    [](const GemmArgs &args) { return !args._ci->has_fp16(); },
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_sgemm_8x12, __fp16, __fp16, __fp16>(args);
                    }
            },
#elif defined(__arm__)
            {
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, __fp16, __fp16, __fp16>(args); }
},
#else // not AArch64 or AArch32
# error Unknown Architecture
#endif
            {
                    GemmMethod::DEFAULT,
                    "",
                    nullptr,
                    nullptr,
                    nullptr,
            }
    };

    template<>
    const GemmImplementation<__fp16, __fp16, __fp16> *gemm_implementation_list<__fp16, __fp16, __fp16>() {
        return gemm_fp16_methods;
    }

/* Explicitly instantiate the external functions for these types. */
    template UniqueGemmCommon<__fp16, __fp16, __fp16>
    gemm<__fp16, __fp16, __fp16, Nothing>(const GemmArgs &args, const Nothing &);

    template bool
    has_opt_gemm<__fp16, __fp16, __fp16, Nothing>(WeightFormat &weight_format, const GemmArgs &args, const Nothing &);

    template KernelDescription get_gemm_method<__fp16, __fp16, __fp16, Nothing>(const GemmArgs &args, const Nothing &);

    template std::vector<KernelDescription>
    get_compatible_kernels<__fp16, __fp16, __fp16, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm

#endif // defined(__aarch64__) && (defined(ENABLE_FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))