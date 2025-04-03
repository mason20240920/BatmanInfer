//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_BI_CPU_RESHAPE_KERNEL_HPP
#define BATMANINFER_BI_CPU_RESHAPE_KERNEL_HPP

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * 张量形状变性接口: 内核接口
             */
            class BICpuReshapeKernel : public BIICpuKernel<BICpuReshapeKernel> {
            public:
                BICpuReshapeKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuReshapeKernel);

                /**
                 * 根据指定参数配置内核
                 * @param src 设置源数据: 数据类型: 全部支持
                 * @param dst 目标张量信息: 数据类型 @p input 相同
                 */
                void configure(const BIITensorInfo *src, BIITensorInfo *dst);

                /**
                 * @brief 动态更新Reshape算子结构
                 * @param dst
                 */
                void dynamic_configure(const BIITensorInfo *dst);

                /**
                 * 检查给定信息是否是合法配置
                 * @param src
                 * @param dst
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst);

                void run_op(BatmanInfer::BIITensorPack &tensors, const BatmanInfer::BIWindow &window,
                            const BatmanInfer::ThreadInfo &info) override;

                const char *name() const override;

                /**
                 * 通过计算最大或压缩窗口并根据孔的存在选择 _reshape_tensor_fn，准备重塑内核以供执行（仅执行一次）
                 * @param tensors 输入和输出张量包
                 */
                void prepare(BIITensorPack &tensors);

                /**
                 * 返回相关内核的最小工作负载大小。
                 * @param platform The CPU platform used to create the context
                 * @param thread_count Number of threads in the execution.
                 * @return Minimum workload size for requested configuration.
                 */
                size_t get_mws(const BatmanInfer::CPUInfo &platform, size_t thread_count) const override;

            private:
                std::function<void(const BIWindow &window, const BIITensor *src, BIITensor *dst)> _reshape_tensor_fn{};
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_RESHAPE_KERNEL_HPP
