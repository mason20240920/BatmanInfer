//
// Created by Mason on 2025/1/14.
//

#ifdef __aarch64__

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s16_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s8_8x12.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_gemm_s8_4x4.hpp"
#include "data/core/neon/kernels/arm_gemm/kernels/a64_interleaved_s8s32_mmla_8x12.hpp"

#include "data/core/neon/kernels/arm_gemm/gemm_implementation.hpp"
#include "data/core/neon/kernels/arm_gemm/gemm_interleaved.hpp"
#include "data/core/neon/kernels/arm_gemm/utils.hpp"

#include <cstdint>
#include <vector>

namespace BatmanGemm {

    static const GemmImplementation<int8_t, int8_t, float, DequantizeFloat> gemm_s8fp32_methods[] =
            {
#ifdef ARM_COMPUTE_ENABLE_SVE
#ifdef ARM_COMPUTE_ENABLE_SME2
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL.hpp",
                        [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme2() && !args._accumulate; },
                        [](const GemmArgs &args, const DequantizeFloat &) { const auto VL = sme::get_vector_length<float>();
                                                                            return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
                        [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL, int8_t, float>(args, dq); }
                    },
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_s8qfp32_mopa_4Vx1VL.hpp",
                        [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme2() && !args._accumulate; },
                        [](const GemmArgs &args, const DequantizeFloat &) { const auto VL = sme::get_vector_length<float>();
                                                                            return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
                        [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL, int8_t, float>(args, dq); }
                    },
                    {
                        GemmMethod::GEMM_INTERLEAVED,
                        "sme2_interleaved_nomerge_s8qfp32_mopa_2Vx2VL.hpp",
                        [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme2() && !args._accumulate; },
                        nullptr,
                        [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL, int8_t, float>(args, dq); }
                    },
#endif // ARM_COMPUTE_ENABLE_SME2
                    GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_s8s32_mmla_8x3VL",
                        [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
                        [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
                        [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, float>(args, qp); }
                    ),
                    GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
                        GemmMethod::GEMM_INTERLEAVED,
                        "sve_interleaved_s8s32_dot_8x3VL",
                        [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sve(); },
                        [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
                        [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, float>(args, qp); }
                    ),
#endif // ARM_COMPUTE_ENABLE_SVE
                    GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_interleaved_s8s32_mmla_8x12",
                            [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
                            [](const GemmArgs &args, const DequantizeFloat &) {
                                return GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, float>::estimate_cycles<int8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const DequantizeFloat &qp) {
                                return new GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, float>(
                                        args, qp);
                            }
                    ),
                    {
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_gemm_s16_8x12",
                            nullptr,
                            [](const GemmArgs &args, const DequantizeFloat &) {
                                return args._ci->get_cpu_model() == BICPUModel::A53 &&
                                       ((args._Msize > 28) || ((args._Msize % 8) > 4));
                            },
                            [](const GemmArgs &args, const DequantizeFloat &qp) {
                                return new GemmInterleavedDequantized<cls_a64_gemm_s16_8x12, int8_t, int8_t, float>(
                                        args, qp);
                            }
                    },
                    GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_gemm_s8_8x12",
                            [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_dotprod(); },
                            [](const GemmArgs &args, const DequantizeFloat &) {
                                return GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t, float>::estimate_cycles<int8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const DequantizeFloat &qp) {
                                return new GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t, float>(args,
                                                                                                                   qp);
                            }
                    ),
                    GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
                            GemmMethod::GEMM_INTERLEAVED,
                            "a64_gemm_s8_4x4",
                            nullptr,
                            [](const GemmArgs &args, const DequantizeFloat &) {
                                return GemmInterleavedDequantized<cls_a64_gemm_s8_4x4, int8_t, int8_t, float>::estimate_cycles<int8_t>(
                                        args);
                            },
                            [](const GemmArgs &args, const DequantizeFloat &qp) {
                                return new GemmInterleavedDequantized<cls_a64_gemm_s8_4x4, int8_t, int8_t, float>(args,
                                                                                                                  qp);
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
    const GemmImplementation<int8_t, int8_t, float, DequantizeFloat> *
    gemm_implementation_list<int8_t, int8_t, float, DequantizeFloat>() {
        return gemm_s8fp32_methods;
    }

    template UniqueGemmCommon<int8_t, int8_t, float>
    gemm<int8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

    template KernelDescription
    get_gemm_method<int8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

    template std::vector<KernelDescription>
    get_compatible_kernels<int8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

} // namespace arm_gemm

#endif // __aarch64__