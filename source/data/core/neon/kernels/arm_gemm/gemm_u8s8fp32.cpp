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

    static const GemmImplementation<uint8_t, int8_t, float, DequantizeFloat> gemm_u8s8fp32_methods[] =
            {
#ifdef BI_COMPUTE_ENABLE_SVE
                    GemmImplementation<uint8_t, int8_t, float, DequantizeFloat>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, float>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, float>(args, qp); }
),
#endif // BI_COMPUTE_ENABLE_SVE
                    GemmImplementation<uint8_t, int8_t, float, DequantizeFloat>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_interleaved_u8s8s32_mmla_8x12",
                            [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
                            [](const GemmArgs &args, const DequantizeFloat &) {
                                return GemmInterleavedDequantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, float>::estimate_cycles<uint8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const DequantizeFloat &qp) {
                                return new GemmInterleavedDequantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, float>(
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
    const GemmImplementation<uint8_t, int8_t, float, DequantizeFloat> *
    gemm_implementation_list<uint8_t, int8_t, float, DequantizeFloat>() {
        return gemm_u8s8fp32_methods;
    }

    template UniqueGemmCommon<uint8_t, int8_t, float>
    gemm<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

    template bool
    has_opt_gemm<uint8_t, int8_t, float, DequantizeFloat>(WeightFormat &weight_format, const GemmArgs &args,
                                                          const DequantizeFloat &os);

    template KernelDescription
    get_gemm_method<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

    template std::vector<KernelDescription>
    get_compatible_kernels<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

} // namespace arm_gemm

#endif // __aarch64__