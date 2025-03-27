//
// Created by Mason on 2025/3/27.
//

#include <cpu/kernels/dynamic_gemm/heuristics/bi_cpu_dynamic_gemm_kernel_heuristics.hpp>

#include <common/bi_registers.hpp>
#include <cpu/kernels/bi_cpu_dynamic_gemm_kernel.hpp>
#include <cpu/kernels/dynamic_gemm/generic/impl.hpp>

#include <algorithm>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace heuristics {
                const BICpuDynamicGemmKernelHeuristics::KernelList BICpuDynamicGemmKernelHeuristics::fp32_kernels{
#if defined(__aarch64__)
                        {"neon_fp32_dynamic_gemm",
                         [](const BIDataTypeISASelectorData &data) {
                             BI_COMPUTE_UNUSED(data);
                             return true;
                         },
                         REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_run),
                         REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_pack_rhs),
                         REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_size_of_packed_rhs),
                         REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_window)
                        },
#endif /* __aarch64__ */
                };

                const BICpuDynamicGemmKernelHeuristics::KernelMap BICpuDynamicGemmKernelHeuristics::kernels{
                        {BIDataType::F32, fp32_kernels},
                };

                void BICpuDynamicGemmKernelHeuristics::choose_kernel(const BIDataTypeISASelectorData &selector) {
                    const auto &klist = kernels.find(selector.dt);
                    BI_COMPUTE_ERROR_ON(klist == kernels.end());

                    for (const auto &uk: klist->second) {
                        if (uk.is_selected(selector)) {
                            _kernel = &uk;
                            return;
                        }
                    }
                }

                BICpuDynamicGemmKernelHeuristics::BICpuDynamicGemmKernelHeuristics(const BIITensorInfo *a,
                                                                                   const BIITensorInfo *b,
                                                                                   const BIITensorInfo *c,
                                                                                   BIITensorInfo *d,
                                                                                   float alpha,
                                                                                   float beta,
                                                                                   const GEMMInfo &gemm_info) {
                    BI_COMPUTE_UNUSED(b);
                    BI_COMPUTE_UNUSED(c);
                    BI_COMPUTE_UNUSED(d);
                    BI_COMPUTE_UNUSED(alpha);
                    BI_COMPUTE_UNUSED(beta);
                    BI_COMPUTE_UNUSED(gemm_info);

                    const BIDataTypeISASelectorData selector{a->data_type(), CPUInfo::get().get_isa()};
                    choose_kernel(selector);
                }

                size_t BICpuDynamicGemmKernelHeuristics::mws() const {
                    return _mws;
                }

                BICpuDynamicGemmKernelHeuristics::KernelPtr BICpuDynamicGemmKernelHeuristics::kernel() const {
                    BI_COMPUTE_ERROR_ON_NULLPTR(_kernel);
                    return _kernel->ukernel;
                }

                BICpuDynamicGemmKernelHeuristics::PackRhsPtr BICpuDynamicGemmKernelHeuristics::pack_rhs() const {
                    BI_COMPUTE_ERROR_ON_NULLPTR(_kernel);
                    return _kernel->pack_rhs;
                }

                BICpuDynamicGemmKernelHeuristics::SizeOfPackedRhsPtr
                BICpuDynamicGemmKernelHeuristics::size_of_packed_rhs() const {
                    BI_COMPUTE_ERROR_ON_NULLPTR(_kernel);
                    return _kernel->size_of_packed_rhs;
                }

                BICpuDynamicGemmKernelHeuristics::GetWindowPtr BICpuDynamicGemmKernelHeuristics::get_window() const {
                    BI_COMPUTE_ERROR_ON_NULLPTR(_kernel);
                    return _kernel->get_window;
                }

                const char *BICpuDynamicGemmKernelHeuristics::name() const {
                    BI_COMPUTE_ERROR_ON_NULLPTR(_kernel);
                    return _kernel->name;
                }

                const BIIScheduler::Hints &BICpuDynamicGemmKernelHeuristics::scheduler_hint() const {
                    return _hint;
                }
            }
        }
    }
}