//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <data/core/helpers/bi_lut_manager.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            class BICpuSoftmaxKernel : public BIICpuKernel<BICpuSoftmaxKernel> {
            private:
                using BISoftmaxKernelPtr = std::add_pointer<void(const BIITensor *,
                                                                 void *const,
                                                                 BIITensor *,
                                                                 float,
                                                                 int,
                                                                 const BIWindow &,
                                                                 const void *)>::type;
            public:
                BICpuSoftmaxKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuSoftmaxKernel);

                /**
                 * 设置输入张量和输出张量
                 * @param src
                 * @param dst
                 * @param beta
                 * @param is_log
                 * @param axis
                 * @param tmp
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst,
                               float beta,
                               bool is_log,
                               int axis,
                               BIITensorInfo *tmp);

                /** Static function to check if given info will lead to a valid configuration
                *
                * Similar to CpuSoftmaxKernel::configure()
                *
                * @return a status
                */
                static BIStatus
                validate(const BIITensorInfo *src, const BIITensorInfo *dst, float beta, int axis, bool is_log,
                         const BIITensorInfo *tmp);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

                struct BISoftmaxKernel {
                    const char *name;
                    const SoftmaxKernelDataTypeISASelectorDataPtr is_selected;
                    BISoftmaxKernelPtr ukernel;
                };

                static const std::vector<BISoftmaxKernel> &get_available_kernels();

            private:
                float _beta{1.0f};
                BISoftmaxKernelPtr _run_method{nullptr};
                std::string _name{};
                int _axis{};
#ifdef __aarch64__
                std::shared_ptr<LookupTable256> _lut{nullptr};
                std::shared_ptr<LookupTable65536> _lut_bf16 = nullptr;
#endif // __aarch64__
            };
        }
    }
}