//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_BI_CPU_ADD_KERNEL_HPP
#define BATMANINFER_BI_CPU_ADD_KERNEL_HPP

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

#include "data/core/bi_tensor_info.hpp"

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            class BICpuAddKernel : public BIICpuKernel<BICpuAddKernel> {
            private:
                using BIAddKernelPtr = std::add_pointer<void(const BIITensor *,
                                                             const BIITensor *, BIITensor *,
                                                             const BIConvertPolicy &,
                                                             const BIWindow &)>::type;

            public:
                struct BIAddKernel {
                    const char *name;
                    const CpuAddKernelDataTypeISASelectorDataPtr is_selected;
                    BIAddKernelPtr ukernel;
                };

                BICpuAddKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuAddKernel);

                void
                configure(const BIITensorInfo *src0,
                          const BIITensorInfo *src1,
                          BIITensorInfo *dst,
                          BIConvertPolicy policy);

                /**
                 * @brief 动态Add函数配置
                 * @param src0
                 * @param src1
                 * @param is_til_mat
                 */
                void dynamic_configure(const BatmanInfer::BIITensorInfo *src0,
                                       const BatmanInfer::BIITensorInfo *src1,
                                       bool is_til_mat = false);

                /**
                 * 静态函数，用于检查给定信息是否会导致有效的配置。
                 * @param src0
                 * @param src1
                 * @param dst
                 * @param policy
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src0,
                                         const BIITensorInfo *src1,
                                         const BIITensorInfo *dst,
                                         BIConvertPolicy policy);

                void run_op(BIITensorPack &tensors,
                            const BIWindow &window,
                            const ThreadInfo &info) override;

                const char *name() const override;

                /**
                 * 返回最小的相对内核的工作负载
                 * @param platform
                 * @param thread_count
                 * @return
                 */
                size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

                static const std::vector<BIAddKernel> &get_available_kernels();

                size_t get_split_dimension() const {
                    return _split_dimension;
                }

            private:
                BIConvertPolicy _policy{};
                BIAddKernelPtr _run_method{nullptr};
                std::string _name{};
                size_t _split_dimension{BIWindow::DimY};
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_ADD_KERNEL_HPP
