//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/experimental/types.hpp>
#include <data/core/bi_tensor_info.hpp>

#include <cpu/bi_i_cpu_kernel.hpp>
#include <cpu/bi_i_cpu_operator.hpp>
#include <cpu/operators/bi_cpu_permute.hpp>

#include <memory>

namespace BatmanInfer {
    namespace cpu {
        class BICpuSoftmaxKernel;

        /** Basic function to compute a SoftmaxLayer and a Log SoftmaxLayer.
         *
         * Softmax is calculated by :
         * @f[ out = exp((x - max(x)) * beta) / sum(exp((x - max(x)) * beta)) @f]
         *
         * Log Softmax is calculated by :
         * @f[ out = (x - max(x) * beta) - log(\sum{e^{x - max(x) * beta}}) @f]
         *
         * This function runs the following function/kernels:
         * -# If axis is not 0:
         * -# @ref CpuPermute
         * -# @ref kernels::CpuSoftmaxKernel
         */
        class BICpuSoftmaxGeneric : public BIICpuOperator {
        public:
            BICpuSoftmaxGeneric();

            /** Set the input and output tensors.
             *
             * @param[in,out] src    Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
             *                       last value of each row to the nearest multiple.
             * @param[out]    dst    Destination tensor ifo. Data types supported: same as @p input.
             * @param[in]     beta   (Optional) A scaling factor for the exponent.
             * @param[in]     axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
             *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
             * @param[in]     is_log True if the operation is log-softmax
             */
            void configure(const BIITensorInfo *src, BIITensorInfo *dst, float beta = 1.0f, int32_t axis = 0,
                           bool is_log = false);

            void dynamic_configure(const BIITensorInfo *src, const BIITensorInfo *dst) const;

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuSoftmaxGeneric::configure()
             *
             * @return a status
             */
            static BIStatus
            validate(const BIITensorInfo *src, const BIITensorInfo *dst, float beta = 1.0f, int32_t axis = 0,
                     bool is_log = false);

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;

            experimental::BIMemoryRequirements workspace() const override;

        private:
            enum InternalTensorIdx {
                TMP = 0,
                PERMUTED_SRC,
                PERMUTED_DST,
                COUNT
            };

            std::unique_ptr<BIICPPKernel> _softmax_kernel;

            BITensorInfo _tmp;

            experimental::BIMemoryRequirements _aux_mem{};

            unsigned int _axis = 0;
        };

    } // namespace cpu
}