//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_CPU_ACTIVATION_KERNEL_HEURISTICS_HPP
#define BATMANINFER_BI_CPU_ACTIVATION_KERNEL_HEURISTICS_HPP

#include <data/core/cpp/bi_i_cpp_kernel.hpp>
#include <data/core/bi_i_tensor_info.hpp>
#include <data/core/bi_window.hpp>
#include <function_info/bi_activationLayerInfo.h>
#include <runtime/bi_i_scheduler.hpp>

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/kernels/bi_cpu_kernel_selection_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace heuristics {
                /**
                 * 为 CPU 上的激活函数运算选择最优的执行策略。通过分析以下因素来决定使用何种实现方式：
                 *
                 * 输入数据的特征（大小、类型等）
                 * 激活函数的类型（ReLU、Sigmoid等）
                 * 硬件特性（CPU架构、指令集等）
                 * 通常会在以下几种实现方式中选择：
                 *
                 * 表法（LUT）
                 *
                 * 使用前面提到的 LUTManager
                 * 适用于计算复杂的激活函数（如 sigmoid、tanh）
                 * 在数值范围有限时效率最高
                 * 直接计算
                 *
                 * 直接使用数学公式计算
                 * 适用于简单的激活函数（如 ReLU）
                 * 当数值范围很大时更合适
                 * SIMD 优化
                 *
                 * 使用 NEON 等 SIMD 指令
                 * 适用于批量数据处理
                 * 考虑数据对齐等因素
                 */
                class BICpuActivationKernelHeuristics {
                public:
                    using BIKernelPtr = std::add_pointer<void(const BIITensor *,
                                                              BIITensor *,
                                                              const BIActivationLayerInfo &,
                                                              const BIWindow &)>::type;
                    struct BIActivationKernel {
                        const char *name{nullptr};
                        const ActivationDataTypeISASelectorDataPtr is_selected{nullptr};
                        BIKernelPtr ukernel{nullptr};
                    };

                    BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuActivationKernelHeuristics);

                    // 默认构造函数和析构
                    BICpuActivationKernelHeuristics() noexcept {};

                    ~BICpuActivationKernelHeuristics() = default;

                    BICpuActivationKernelHeuristics(const BIITensorInfo *src,
                                                    const BIITensorInfo *dst,
                                                    const BIActivationLayerInfo &activation_info);

                    /**
                     * 最小工作负载
                     * @return
                     */
                    size_t mws() const;

                    const BIWindow &window() const;

                    /**
                     * 返回运行的内核
                     * @return
                     */
                    const BIActivationKernel *kernel();

                    /**
                     * 返回调度提示，例如要拆分的维度。
                     * @return 一个 @ref IScheduler::Hints 实例来描述调度提示。
                     */
                    const BIIScheduler::Hints &scheduler_hint() const;

                private:

                    /**
                     * 选择一个内核运行并将其保存到_内核数据成员中
                     *
                     * @param selector
                     */
                    void choose_kernel(ActivationDataTypeISASelectorData &selector);

                private:
                    size_t _mws{BIICPPKernel::default_mws};
                    BIWindow _window{};
                    const BIActivationKernel *_kernel{nullptr};
                    BIIScheduler::Hints _hint{BIWindow::DimY};
                };
            }
        }
    }
}

#endif //BATMANINFER_BI_CPU_ACTIVATION_KERNEL_HEURISTICS_HPP
