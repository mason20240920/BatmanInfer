//
// Created by Mason on 2025/1/14.
//

#ifdef __aarch64__

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_u8s8qa_dot_4x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_u8s8qa_mmla_4x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_u8s8s32_dot_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_hybrid_u8s8s32_mmla_6x16.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_interleaved_u8s8s32_mmla_8x12.hpp"

#include "data/core/neon/kernels/arm_gemm/gemm_hybrid_indirect.hpp"
#include "data/core/neon/kernels/arm_gemm/gemm_hybrid_quantized.hpp"
#include "data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp"
#include "data/core/neon/kernels/arm_gemm/gemv_pretransposed.hpp"
#include "data/core/neon/kernels/arm_gemm/quantize_wrapper.hpp"
#include "data/core/neon/kernels/arm_gemm/utils.hpp"

namespace BatmanGemm {

    static const GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32> gemm_q8_mixed_methods[] =
            {
#ifdef ARM_COMPUTE_ENABLE_SVE
                    GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_u8s8qa_mmla_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return quant_hybrid_asymmetric(qp) && args._ci->has_sve2() && args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8s8qa_mmla_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8s8qa_mmla_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_svei8mm() && (args._Ksize>8); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_hybrid_u8s8s32_mmla_6x4VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8s8s32_mmla_6x4VL, uint8_t, int8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8s8s32_mmla_6x4VL, uint8_t, int8_t, uint8_t, Requantize32, true>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_u8s8qa_dot_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sve2() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8s8qa_dot_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8s8qa_dot_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>(args, qp); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
                    GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_u8s8qa_mmla_4x16",
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return args._ci->has_i8mm() && quant_hybrid_asymmetric(qp);
                            },
                            [](const GemmArgs &args, const Requantize32 &) {
                                return GemmHybridIndirect<cls_a64_hybrid_u8s8qa_mmla_4x16, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return new GemmHybridIndirect<cls_a64_hybrid_u8s8qa_mmla_4x16, uint8_t, int8_t, uint8_t, Requantize32>(
                                        args, qp);
                            }
                    ),
                    GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_interleaved_u8s8s32_mmla_8x12",
                            [](const GemmArgs &args, const Requantize32 &) {
                                return args._ci->has_i8mm() && (args._Ksize > 8);
                            },
                            [](const GemmArgs &args, const Requantize32 &) {
                                return GemmInterleavedQuantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, uint8_t>::estimate_cycles<uint8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return new GemmInterleavedQuantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, uint8_t>(
                                        args, qp);
                            }
                    ),
                    GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_hybrid_u8s8s32_mmla_6x16",
                            [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_i8mm(); },
                            [](const GemmArgs &args, const Requantize32 &) {
                                return GemmHybridIndirect<cls_a64_hybrid_u8s8s32_mmla_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return new GemmHybridIndirect<cls_a64_hybrid_u8s8s32_mmla_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>(
                                        args, qp);
                            }
                    ),
                    GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_u8s8qa_dot_4x16",
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return args._ci->has_svei8mm() && quant_hybrid_asymmetric(qp);
                            },
                            [](const GemmArgs &args, const Requantize32 &) {
                                return GemmHybridIndirect<cls_a64_hybrid_u8s8qa_dot_4x16, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return new GemmHybridIndirect<cls_a64_hybrid_u8s8qa_dot_4x16, uint8_t, int8_t, uint8_t, Requantize32>(
                                        args, qp);
                            }
                    ),
                    GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
                            GemmMethod::GEMM_HYBRID,
                            "a64_hybrid_u8s8s32_dot_6x16",
                            [](const GemmArgs &args, const Requantize32 &) {
                                return args._ci->has_dotprod() && args._ci->has_i8mm();
                            },
                            [](const GemmArgs &args, const Requantize32 &) {
                                return GemmHybridIndirect<cls_a64_hybrid_u8s8s32_dot_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const Requantize32 &qp) {
                                return new GemmHybridIndirect<cls_a64_hybrid_u8s8s32_dot_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>(
                                        args, qp);
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
    const GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32> *
    gemm_implementation_list<uint8_t, int8_t, uint8_t, Requantize32>() {
        return gemm_q8_mixed_methods;
    }

    template UniqueGemmCommon<uint8_t, int8_t, uint8_t>
    gemm<uint8_t, int8_t, uint8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);

    template bool
    has_opt_gemm<uint8_t, int8_t, uint8_t, Requantize32>(WeightFormat &weight_format, const GemmArgs &args,
                                                         const Requantize32 &);

    template KernelDescription
    get_gemm_method<uint8_t, int8_t, uint8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);

    template std::vector<KernelDescription>
    get_compatible_kernels<uint8_t, int8_t, uint8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);

} // namespace arm_gemm

#endif // __aarch64__