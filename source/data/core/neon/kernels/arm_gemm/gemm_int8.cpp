//
// Created by Mason on 2025/1/14.
//

#ifdef __aarch64__

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <cpu/kernels/assembly/bi_gemm_common.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid_indirect.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_batched.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_pretransposed.hpp>

#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s16_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s8_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s8_4x4.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_s8s32_dot_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_s8s32_mmla_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_smallK_hybrid_s8s32_dot_6x4.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_smallK_hybrid_s8s32_dot_8x4.hpp"

namespace BatmanGemm {

    static const GemmImplementation<int8_t, int8_t, int32_t> gemm_s8_methods[] = {
#ifdef BI_COMPUTE_ENABLE_SVE
#ifdef BI_COMPUTE_ENABLE_SME2
            // SME kernels
            {
                GemmMethod::GEMM_INTERLEAVED,
                "sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL",
                [](const GemmArgs &args) { return args._ci->has_sme2(); },
                [](const GemmArgs &args) { const auto VL = sme::get_vector_length<int32_t>();
                                           return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
                [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL, int8_t, int32_t>(args); }
            },
            {
                GemmMethod::GEMM_INTERLEAVED,
                "sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL",
                [](const GemmArgs &args) { return args._ci->has_sme2(); },
                [](const GemmArgs &args) { const auto VL = sme::get_vector_length<int32_t>();
                                           return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
                [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL, int8_t, int32_t>(args); }
            },
            {
                GemmMethod::GEMM_INTERLEAVED,
                "sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL",
                [](const GemmArgs &args) { return args._ci->has_sme2(); },
                nullptr,
                [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL, int8_t, int32_t>(args); }
            },
#endif // BI_COMPUTE_ENABLE_SME2
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                GemmMethod::GEMM_HYBRID,
                "sve_hybrid_s8s32_mmla_6x4VL",
                [](const GemmArgs &args) { return args._ci->has_svei8mm(); },
                [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_s8s32_mmla_6x4VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
                [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_s8s32_mmla_6x4VL, int8_t, int8_t, int32_t>(args); }
            ),
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                GemmMethod::GEMM_INTERLEAVED,
                "sve_interleaved_s8s32_mmla_8x3VL",
                [](const GemmArgs &args) { return args._ci->has_svei8mm() && (args._Ksize>8); },
                [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
                [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, int32_t>(args); }
            ),
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                GemmMethod::GEMM_HYBRID,
                "sve_hybrid_s8s32_dot_6x4VL",
                [](const GemmArgs &args) { return args._ci->has_sve() && args._Ksize>=16; },
                [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_s8s32_dot_6x4VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
                [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_s8s32_dot_6x4VL, int8_t, int8_t, int32_t>(args); }
            ),
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                GemmMethod::GEMM_INTERLEAVED,
                "sve_interleaved_s8s32_dot_8x3VL",
                [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize>4); },
                [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
                [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, int32_t>(args); }
            ),
#endif // BI_COMPUTE_ENABLE_SVE
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_interleaved_s8s32_mmla_8x12",
                    [](const GemmArgs &args) { return args._ci->has_i8mm() && (args._Ksize > 8); },
                    [](const GemmArgs &args) {
                        return GemmInterleaved<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, int32_t>(args);
                    }
            ),
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                    GemmMethod::GEMM_HYBRID,
                    "a64_hybrid_s8s32_mmla_6x16",
                    [](const GemmArgs &args) { return args._ci->has_i8mm(); },
                    [](const GemmArgs &args) {
                        return GemmHybridIndirect<cls_a64_hybrid_s8s32_mmla_6x16, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmHybridIndirect<cls_a64_hybrid_s8s32_mmla_6x16, int8_t, int8_t, int32_t>(args);
                    }
            ),
            {
                    GemmMethod::GEMM_HYBRID,
                    "a64_smallK_hybrid_s8s32_dot_8x4",
                    [](const GemmArgs &args) {
                        return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize <= 32) &&
                               !args._indirect_input && !args._accumulate;
                    },
                    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
                    [](const GemmArgs &args) {
                        return new GemmHybrid<cls_a64_smallK_hybrid_s8s32_dot_8x4, int8_t, int32_t>(args);
                    }
            },
            {
                    GemmMethod::GEMM_HYBRID,
                    "a64_smallK_hybrid_s8s32_dot_6x4",
                    [](const GemmArgs &args) {
                        return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize > 32) &&
                               (args._Ksize <= 64) && !args._indirect_input && !args._accumulate;
                    },
                    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
                    [](const GemmArgs &args) {
                        return new GemmHybrid<cls_a64_smallK_hybrid_s8s32_dot_6x4, int8_t, int32_t>(args);
                    }
            },
            {
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_gemm_s16_8x12",
                    nullptr,
                    [](const GemmArgs &args) {
                        return args._ci->get_cpu_model() == BICPUModel::A53 &&
                               ((args._Msize > 28) || ((args._Msize % 8) > 4));
                    },
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_gemm_s16_8x12, int8_t, int8_t, int32_t>(args);
                    },
            },
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(

                    GemmMethod::GEMM_HYBRID,
                    "a64_hybrid_s8s32_dot_6x16",
                    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
                    [](const GemmArgs &args) {
                        return GemmHybridIndirect<cls_a64_hybrid_s8s32_dot_6x16, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmHybridIndirect<cls_a64_hybrid_s8s32_dot_6x16, int8_t, int8_t, int32_t>(args);
                    }
            ),
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_gemm_s8_8x12",
                    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
                    [](const GemmArgs &args) {
                        return GemmInterleaved<cls_a64_gemm_s8_8x12, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_gemm_s8_8x12, int8_t, int8_t, int32_t>(args);
                    }
            ),
            GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
                    GemmMethod::GEMM_INTERLEAVED,
                    "a64_gemm_s8_4x4",
                    nullptr,
                    [](const GemmArgs &args) {
                        return GemmInterleaved<cls_a64_gemm_s8_4x4, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(
                                args);
                    },
                    [](const GemmArgs &args) {
                        return new GemmInterleaved<cls_a64_gemm_s8_4x4, int8_t, int8_t, int32_t>(args);
                    }
            ),

            {
                    GemmMethod::DEFAULT,
                    "",
                    nullptr,
                    nullptr,
                    nullptr
            }
    };

    template<>
    const GemmImplementation<int8_t, int8_t, int32_t> *gemm_implementation_list<int8_t, int8_t, int32_t>() {
        return gemm_s8_methods;
    }

/* Explicitly instantiate the external functions for these types. */
    template UniqueGemmCommon<int8_t, int8_t, int32_t>
    gemm<int8_t, int8_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);

    template bool
    has_opt_gemm<int8_t, int8_t, int32_t, Nothing>(WeightFormat &weight_format, const GemmArgs &args, const Nothing &);

    template KernelDescription get_gemm_method<int8_t, int8_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);

    template std::vector<KernelDescription>
    get_compatible_kernels<int8_t, int8_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm

#endif // __aarch64__