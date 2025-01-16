//
// Created by Mason on 2025/1/16.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Interface for an element-wise operation kernel
             *
             * Element-wise operation is computed by:
             * @f[ dst(x,y) = OP(src0(x,y), src1(x,y))@f]
             *
             */
            template<class Derived>
            class BICpuElementwiseKernel : public BIICpuKernel<Derived> {
            private:
                using BIElementwiseKernelPtr = std::add_pointer<void(const BIITensor *, const BIITensor *, BIITensor *,
                                                                     const BIWindow &)>::type;

            public:
                BICpuElementwiseKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuElementwiseKernel);

                using BIElementwiseFunction = void(const BIITensor *, const BIITensor *, BIITensor *, const BIWindow &);

                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

                struct ElementwiseKernel {
                    const char *name;
                    const ElementwiseDataTypeISASelectorPtr is_selected;
                    BIElementwiseKernelPtr ukernel;
                };

            protected:
                /** Validate the argument passed to the kernel
                 *
                 * @param[in] src0 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
                 * @param[in] src1 Second tensor input. Data types supported: Same as @p src0.
                 * @param[in] dst  Output tensor. Data types supported: Dependent on subclass.
                 */
                static BIStatus
                validate_arguments_common(const BIITensorInfo &src0,
                                          const BIITensorInfo &src1,
                                          const BIITensorInfo &dst);

            protected:
                BIElementwiseKernelPtr _run_method{nullptr};
                std::string _name{};
            };

            class BICpuArithmeticKernel : public BICpuElementwiseKernel<BICpuArithmeticKernel> {
            public:
                BICpuArithmeticKernel() = default;

                /** Configure kernel
                 *
                 * @param[in]  op   Arithmetic operation to be executed.
                 * @param[in]  src0 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
                 * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
                 * @param[out] dst  Output tensor info. Data types supported: Same as @p src0.
                 */
                void
                configure(ArithmeticOperation op,
                          const BIITensorInfo *src0,
                          const BIITensorInfo *src1,
                          BIITensorInfo *dst);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuArithmeticKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(ArithmeticOperation op,
                         const BIITensorInfo *src0,
                         const BIITensorInfo *src1,
                         const BIITensorInfo *dst);

                static const std::vector<BICpuElementwiseKernel<BICpuArithmeticKernel>::ElementwiseKernel> &
                get_available_kernels();

                /** Return minimum workload size of the relevant kernel
                 *
                 * @param[in] platform     The CPU platform used to create the context.
                 * @param[in] thread_count Number of threads in the execution.
                 *
                 * @return[out] mws Minimum workload size for requested configuration.
                 */
                size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

            protected:
                /** Common configure function for element-wise operators with no additional options (e.g. Min, Max, SquaredDiff)
                 */
                void configure_common(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

                // Inherited methods overridden:
                static BIStatus
                validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1, const BIITensorInfo &dst);

                ArithmeticOperation _op{};
            };


            class BICpuDivisionKernel : public BICpuArithmeticKernel {
            public:
                BICpuDivisionKernel() = default;

                /** Configure kernel
                 *
                 * @param[in]  src0 First tensor input info. Data types supported: S32/F16/F32.
                 * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
                 * @param[out] dst  Output tensor info. Data types supported: Same as @p src0.
                 */
                void configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuDivisionKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);

                /** Return minimum workload size of the relevant kernel
                 *
                 * @param[in] platform     The CPU platform used to create the context.
                 * @param[in] thread_count Number of threads in the execution.
                 *
                 * @return[out] mws Minimum workload size for requested configuration.
                 */
                size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

            protected:
                // Inherited methods overridden:
                static BIStatus
                validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1, const BIITensorInfo &dst);
            };

            class CpuPowerKernel : public BICpuArithmeticKernel {
            public:
                CpuPowerKernel() = default;

                /** Configure kernel
                 *
                 * @param[in]  src0 First tensor input info. Data types supported: F16/F32.
                 * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
                 * @param[out] dst  Output tensor info. Data types supported: Same as @p src0.
                 */
                void configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuPowerKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);

            protected:
                // Inherited methods overridden:
                static BIStatus
                validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1, const BIITensorInfo &dst);
            };

            class BICpuComparisonKernel : public BICpuElementwiseKernel<BICpuComparisonKernel> {
            public:
                BICpuComparisonKernel() = default;

                /** Configure kernel
                 *
                 * @param[in]  op   Comparison operation to be executed.
                 * @param[in]  src0 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
                 * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
                 * @param[out] dst  Output tensor info. Data types supported: U8.
                 */
                void
                configure(ComparisonOperation op, const BIITensorInfo *src0, const BIITensorInfo *src1,
                          BIITensorInfo *dst);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuComparisonKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(ComparisonOperation op, const BIITensorInfo *src0, const BIITensorInfo *src1,
                         const BIITensorInfo *dst);

                static const std::vector<BICpuElementwiseKernel<BICpuComparisonKernel>::ElementwiseKernel> &
                get_available_kernels();

            protected:
                /** Common configure function for element-wise operators with no additional options (e.g. Min, Max, SquaredDiff)
                 */
                void configure_common(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

                // Inherited methods overridden:
                static BIStatus
                validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1, const BIITensorInfo &dst);

            private:
                /** Function to get the micro kernel implementation
                 *
                 * @param[in] src0 First input tensor information
                 * @param[in] src1 Second input tensor information
                 * @param[in] dst  Output tensor information
                 *
                 * @return the function instance for the micro kernel
                 */

                ComparisonOperation _op{};
            };

        } // namespace kernels
    }
}