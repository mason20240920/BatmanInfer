//
// Created by Mason on 2025/1/14.
//

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <cpu/kernels/assembly/bi_gemm_common.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_hybrid_indirect.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>
#include <data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_batched.hpp>
#include <data/core/neon/kernels/arm_gemm/gemv_pretransposed.hpp>

#include "data/core/neon/kernels/arm_gemm/kernels/a32_sgemm_8x6.hpp"

#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffhybrid_fp32_mla_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffhybrid_fp32bf16fp32_mmla_4x24.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffhybrid_fp32bf16fp32_mmla_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_ffinterleaved_fp32_mla_8x12.hpp"

#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_fp32bf16fp32_mmla_4x24.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_fp32bf16fp32_mmla_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_fp32_mla_4x24.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_fp32_mla_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_fp32_mla_8x4.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_sgemm_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_sgemm_8x6.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_smallK_hybrid_fp32_mla_6x4.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_smallK_hybrid_fp32_mla_8x4.hpp"

namespace BatmanGemm {

    static const GemmImplementation<float, float, float> gemm_fp32_methods[] =
            {
                    // GEMV cases - starting with 'gemv_batched' wrapper to turn batched GEMV into GEMM.
                    {
                            GemmMethod::GEMV_BATCHED,
                            "gemv_batched",
                            [](const GemmArgs &args) {
                                return args._Msize == 1 && args._nbatches > 1 && !args._indirect_input;
                            },
                            nullptr,
                            [](const GemmArgs &args) { return new GemvBatched<float, float>(args); }
                    },
#ifdef __aarch64__
#ifdef BI_COMPUTE_ENABLE_BF16
                    // "fast mode" (BF16) kernels
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_interleaved_bf16fp32_mmla_8x12",
                            [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, float, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, float, float, float>(
                                        args);
                            }
                    ),

                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_fp32bf16fp32_mmla_6x16",
                            [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_6x16, float, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_6x16, float, float, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_fp32bf16fp32_mmla_4x24",
                            [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_4x24, float, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_4x24, float, float, float>(
                                        args);
                            }
                    ),
#endif // BI_COMPUTE_ENABLE_BF16
#ifdef BI_COMPUTE_ENABLE_SVE
#ifdef BI_COMPUTE_ENABLE_SME2
                    // SME kernels
                    {
                        GemmMethod::GEMM_HYBRID,
                        "sme2_gemv_fp32bf16fp32_dot_16VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
                        nullptr,
                        [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp32bf16fp32_dot_16VL, float, float>(args); }
                    },
                    {
                        GemmMethod::GEMM_HYBRID,
                        "sme2_gemv_fp32_mla_16VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
                        nullptr,
                        [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp32_mla_16VL, float, float>(args); }
                    },
#ifdef BI_COMPUTE_ENABLE_BF16
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && !args._accumulate; },
                        [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                                   return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL, float, float>(args); }
                    },
#endif // BI_COMPUTE_ENABLE_BF16
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_fp32_mopa_1VLx4VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2() && !args._accumulate; },
                        [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                                   return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp32_mopa_1VLx4VL, float, float>(args); }
                    },
#ifdef BI_COMPUTE_ENABLE_BF16
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && !args._accumulate; },
                        [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                                   return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL, float, float>(args); }
                    },
#endif // BI_COMPUTE_ENABLE_BF16
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_fp32_mopa_4VLx1VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2() && !args._accumulate; },
                        [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                                                   return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp32_mopa_4VLx1VL, float, float>(args); }
                    },
#ifdef BI_COMPUTE_ENABLE_BF16
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && !args._accumulate; },
                        nullptr,
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL, float, float>(args); }
                    },
#endif // BI_COMPUTE_ENABLE_BF16
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_fp32_mopa_2VLx2VL",
                        [](const GemmArgs &args) { return args._ci->has_sme2() && !args._accumulate; },
                        nullptr,
                        [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp32_mopa_2VLx2VL, float, float>(args); }
                    },
#endif // BI_COMPUTE_ENABLE_SME2
#ifdef BI_COMPUTE_ENABLE_BF16
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_bf16fp32_mmla_8x3VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, float, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, float, float, float>(args); }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_hybrid_fp32bf16fp32_mmla_6x4VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_6x4VL, float, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_6x4VL, float, float, float>(args); }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_hybrid_fp32bf16fp32_mmla_4x6VL",
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_4x6VL, float, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_4x6VL, float, float, float>(args); }
                    ),
#endif // BI_COMPUTE_ENABLE_BF16
#ifdef BI_COMPUTE_ENABLE_SVEF32MM
                    // MMLA next due to higher throughput (which is SVE only)
                    // Prefer this in all cases, except if fast mode is requested and BF16 is available.
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_fp32_mmla_8x3VL",
                        [](const GemmArgs &args) { return args._ci->has_svef32mm() && (args._Ksize>4); },
                        [](const GemmArgs &args) { return !(args._fast_mode && args._ci->has_svebf16()); },
                        [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mmla_8x3VL, float, float, float>(args); }
                    },
#endif // BI_COMPUTE_ENABLE_SVEF32MM
                    // SVE kernels
                    {
                        GemmMethod::GEMM_HYBRID,
                        "sve_hybrid_fp32_mla_8x1VL",
                        [](const GemmArgs &args) { return args._ci->has_sve(); },
                        [](const GemmArgs &args) { return (args._Nsize < 12); },
                        [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_8x1VL, float, float, float>(args); }
                    },
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_hybrid_fp32_mla_6x4VL",
                        [](const GemmArgs &args) { return args._ci->has_sve(); },
                        [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float, float>(args); }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_fp32_mla_8x3VL",
                        [](const GemmArgs &args) { return args._ci->has_sve(); },
                        [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float, float>(args); }
                    ),
#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#ifdef BI_COMPUTE_ENABLE_BF16
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
                        KernelWeightFormat::VL2VL_BL64_BF16,
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, float, float>(args); }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_ffhybrid_fp32bf16fp32_mmla_4x6VL",
                        KernelWeightFormat::VL2VL_BL64_BF16,
                        [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
                        [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32bf16fp32_mmla_4x6VL, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32bf16fp32_mmla_4x6VL, float, float>(args); }
                    ),
#endif
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_ffinterleaved_fp32_mla_8x3VL",
                        KernelWeightFormat::VL1VL_BL32,
                        [](const GemmArgs &args) { return args._ci->has_sve(); },
                        [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp32_mla_8x3VL, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp32_mla_8x3VL, float, float>(args); }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                        GemmMethod::GEMM_HYBRID,
                        "sve_ffhybrid_fp32_mla_6x4VL",
                        KernelWeightFormat::VL1VL_BL32,
                        [](const GemmArgs &args) { return args._ci->has_sve(); },
                        [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32_mla_6x4VL, float, float>::estimate_cycles<float>(args); },
                        [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32_mla_6x4VL, float, float>(args); }
                    ),
#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // BI_COMPUTE_ENABLE_SVE
// Cortex-A35 specific kernel - use for any problem on A35, and never in any other cases.
                    {
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_sgemm_8x6",
                            nullptr,
                            [](const GemmArgs &args) { return args._ci->get_cpu_model() == BICPUModel::A35; },
                            [](const GemmArgs &args) {
                                return new GemmInterleaved<cls_a64_sgemm_8x6, float, float, float>(args);
                            }
                    },
// Arm® Neon™ hybrid methods
                    {
                            GemmMethod::GEMM_HYBRID,
                            "a64_smallK_hybrid_fp32_mla_8x4",
                            [](const GemmArgs &args) {
                                return args._Ksize <= 8 && (args._Nsize % 4) == 0 && !args._indirect_input &&
                                       !args._accumulate;
                            },
                            nullptr,
                            [](const GemmArgs &args) {
                                return new GemmHybrid<cls_a64_smallK_hybrid_fp32_mla_8x4, float, float>(args);
                            }
                    },
                    {
                            GemmMethod::GEMM_HYBRID,
                            "a64_smallK_hybrid_fp32_mla_6x4",
                            [](const GemmArgs &args) {
                                return (args._Ksize > 8 && args._Ksize <= 16) && (args._Nsize % 4) == 0 &&
                                       !args._indirect_input && !args._accumulate;
                            },
                            nullptr,
                            [](const GemmArgs &args) {
                                return new GemmHybrid<cls_a64_smallK_hybrid_fp32_mla_6x4, float, float>(args);
                            }
                    },
                    {
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_fp32_mla_8x4",
                            nullptr,
                            [](const GemmArgs &args) { return (args._Nsize < 12); },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_8x4, float, float, float>(args);
                            }
                    },
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_fp32_mla_4x24",
                            nullptr,
                            [](const GemmArgs &args) {
                                return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_4x24, float, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_4x24, float, float, float>(args);
                            }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_fp32_mla_6x16",
                            nullptr,
                            [](const GemmArgs &args) {
                                return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float, float>(args);
                            }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_sgemm_8x12",
                            nullptr,
                            [](const GemmArgs &args) {
                                return GemmInterleaved<cls_a64_sgemm_8x12, float, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleaved<cls_a64_sgemm_8x12, float, float, float>(args);
                            }
                    ),
#ifdef BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#ifdef BI_COMPUTE_ENABLE_BF16
                    // "fast mode" (BF16) kernels
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_ffinterleaved_bf16fp32_mmla_8x12",
                            KernelWeightFormat::VL256_BL64_BF16,
                            [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, float, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_ffhybrid_fp32bf16fp32_mmla_4x24",
                            KernelWeightFormat::VL256_BL64_BF16,
                            [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_4x24, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_4x24, float, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_ffhybrid_fp32bf16fp32_mmla_6x16",
                            KernelWeightFormat::VL256_BL64_BF16,
                            [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
                            [](const GemmArgs &args) {
                                return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_6x16, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_6x16, float, float>(
                                        args);
                            }
                    ),

#endif // BF16
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_ffinterleaved_fp32_mla_8x12",
                            KernelWeightFormat::VL128_BL32,
                            nullptr,
                            [](const GemmArgs &args) {
                                return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp32_mla_8x12, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp32_mla_8x12, float, float>(
                                        args);
                            }
                    ),
                    GemmImplementation<float, float, float>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_ffhybrid_fp32_mla_6x16",
                            KernelWeightFormat::VL128_BL32,
                            nullptr,
                            [](const GemmArgs &args) {
                                return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32_mla_6x16, float, float>::estimate_cycles<float>(
                                        args);
                            },
                            [](const GemmArgs &args) {
                                return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32_mla_6x16, float, float>(
                                        args);
                            }
                    ),
#endif // BI_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // __aarch64__

#ifdef __arm__
                    {
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, float, float, float>(args); }
},
#endif // __arm__
                    {
                            GemmMethod::DEFAULT,
                            "",
                            nullptr,
                            nullptr,
                            nullptr
                    }
            };

/* Templated function to return this list. */
    template<>
    const GemmImplementation<float, float, float> *gemm_implementation_list<float, float, float>() {
        return gemm_fp32_methods;
    }

/* Explicitly instantiate the external functions for these types. */
    template UniqueGemmCommon<float, float, float>
    gemm<float, float, float, Nothing>(const GemmArgs &args, const Nothing &);

    template bool
    has_opt_gemm<float, float, float, Nothing>(WeightFormat &weight_format, const GemmArgs &args, const Nothing &);

    template KernelDescription get_gemm_method<float, float, float, Nothing>(const GemmArgs &args, const Nothing &);

    template std::vector<KernelDescription>
    get_compatible_kernels<float, float, float, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm