//
// Created by Mason on 2025/3/27.
//

#include <cpu/operators/bi_cpu_dynamic_gemm.hpp>

#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <cpu/kernels/bi_cpu_dynamic_gemm_kernel.hpp>

using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    namespace cpu {
        void BICpuDynamicGemm::configure(const BIITensorInfo *a,
                                         const BIITensorInfo *b,
                                         const BIITensorInfo *c,
                                         BIITensorInfo *d,
                                         float alpha,
                                         float beta,
                                         const GEMMInfo &gemm_info) {
            BI_COMPUTE_ERROR_THROW_ON(BICpuDynamicGemm::validate(a, b, c, d, alpha, beta, gemm_info));
            BI_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, gemm_info);

            _kernel = std::make_unique<kernels::BICpuDynamicGemmKernel>();
            _kernel->configure(a, b, c, d, alpha, beta, Count, gemm_info);

            _reshape_b_and_c_only_on_first_run = b->are_values_constant() && c->are_values_constant();
        }

        BIStatus BICpuDynamicGemm::validate(const BIITensorInfo *a,
                                            const BIITensorInfo *b,
                                            const BIITensorInfo *c,
                                            const BIITensorInfo *d,
                                            float alpha,
                                            float beta,
                                            const GEMMInfo &gemm_info) {
            return kernels::BICpuDynamicGemmKernel::validate(a, b, c, d, alpha, beta, gemm_info);
        }

        void BICpuDynamicGemm::run(BIITensorPack &tensors) {
            BI_COMPUTE_EXIT_ON_MSG(tensors.empty(), "No inputs provided");

            kernels::BICpuDynamicGemmKernel *dynamic_gemm = _kernel.get();
            dynamic_gemm->prepare(tensors, _reuse_b);

            if (_reshape_b_and_c_only_on_first_run) {
                _reuse_b = true;
            }

            BIWindow window = dynamic_gemm->window();
            auto split_dimensions = dynamic_gemm->get_split_dimension_hint();

            BINEScheduler::get().schedule_op(_kernel.get(), split_dimensions, window, tensors);
        }

        const experimental::BIMemoryRequirements &
        BICpuDynamicGemm::workspace_dynamic(const BIITensorPack &tensors) const {
            BI_COMPUTE_ERROR_ON(tensors.empty());
            // Update memory requirements with those from the kernel.
            _aux_mem.reserve(Count + kernels::BICpuDynamicGemmKernel::max_workspace_count());
            _aux_mem.resize(Count);

            for (BIMemoryInfo mi: _kernel->workspace(tensors)) {
                _aux_mem.push_back(mi);
            }

            return _aux_mem;
        }

    }
}