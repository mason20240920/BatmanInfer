//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BI_CPU_ACTIVATION_HPP
#define BATMANINFER_BI_CPU_ACTIVATION_HPP

#include <function_info/bi_activationLayerInfo.h>

#include <cpu/bi_i_cpu_operator.hpp>

#include <data/core/bi_tensor_info.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * 基本的函数: 运行@ref kernels::CpuActivationKernel
         */
        class BICpuActivation : public BIICpuOperator {
        public:
            /** Configure operator for a given list of arguments
            *
            * @param[in]  input           Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
            * @param[out] output          Destination tensor info. Data type supported: same as @p src
            * @param[in]  activation_info Activation layer parameters.
            */
            void configure(const BIITensorInfo *input,
                           BIITensorInfo *output,
                           const BIActivationLayerInfo &activation_info);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuActivation::configure()
             *
             * @return a status
             */
            static BIStatus
            validate(const BIITensorInfo *input,
                     const BIITensorInfo *output, const BIActivationLayerInfo &act_info);

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;

            void dynamic_change_win(const BIITensorInfo *input);
        };
    }
}

#endif //BATMANINFER_BI_CPU_ACTIVATION_HPP
