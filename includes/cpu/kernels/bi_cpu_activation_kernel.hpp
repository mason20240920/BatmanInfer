//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BI_KERNELS_CPU_ACTIVATION_KERNEL_HPP
#define BATMANINFER_BI_KERNELS_CPU_ACTIVATION_KERNEL_HPP

#include <function_info/bi_activationLayerInfo.h>

#include "data/core/common/bi_core_common_macros.hpp"
#include <data/core/helpers/bi_lut_manager.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>
#include <cpu/kernels/activation/heuristics/bi_cpu_activation_kernel_heuristics.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            class BICpuActivationKernel : public BIICPPKernel {
            private:
                using BIActivationKernelPtr = heuristics::BICpuActivationKernelHeuristics::BIKernelPtr;

            public:
                BICpuActivationKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuActivationKernel);

                /** Configure kernel for a given list of arguments
                 *
                 * @note If the output tensor is a nullptr, the activation function will be performed in-place
                 *
                 * @param[in, out] src             Source tensor info. In case of @p dst tensor = nullptr, this tensor will store the result
                 *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
                 * @param[out]     dst             Destination tensor info. Data type supported: same as @p src
                 * @param[in]      activation_info Activation layer information.
                 */
                void configure(const BIITensorInfo *src, BIITensorInfo *dst, BIActivationLayerInfo activation_info);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuActivationKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src, const BIITensorInfo *dst, const BIActivationLayerInfo &act_info);

                /**
                 * @brief 动态修改窗口
                 * @param src
                 */
                void dynamic_change_win(const BIITensorInfo *src);

                /** Return minimum workload size of the relevant kernel
                 *
                 * @param[in] platform     The CPU platform used to create the context.
                 * @param[in] thread_count Number of threads in the execution.
                 *
                 * @return[out] small_network_mws          Minimum workload size for requsted configuration.
                 */
                size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

                /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
                 *
                 * @return The split dimension hint.
                 */
                size_t get_split_dimension_hint() const {
                    return _heuristics.scheduler_hint().split_dimension();
                }

            private:
                BIActivationLayerInfo _act_info{};
                std::string _name{};
                heuristics::BICpuActivationKernelHeuristics _heuristics{};
            };
        }
    }
}

#endif //BATMANINFER_BI_KERNELS_CPU_ACTIVATION_KERNEL_HPP
