//
// Created by Mason on 2025/1/16.
//

#pragma once

#include <cpu/bi_i_cpu_operator.hpp>
#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        class BICpuElementwiseBase : public BIICpuOperator {
        public:
            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;
        };

        /** Class to run @ref cpu::kernels::CpuArithmeticKernel except for division and power
         *
         * @note Max/Min/Squared difference supports input data type of QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32
         * @note PRelu supports inpute data type of QASYMM8/QASYMM8_SIGNED/F16/F32.
         */
        template<ArithmeticOperation op>
        class BICpuElementwiseArithmetic : public BICpuElementwiseBase {
        public:
            /** Configure the operator
             *
             * @param[in]  src0 The first source tensor information.
             * @param[in]  src1 The second source tensor information. With PRelu, this is used as alpha tensor.
             * @param[out] dst  The output tensor information.
             */
            void configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref BICpuElementwiseArithmetic::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);
        };

/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for maximum operation */
        using BICpuElementwiseMax = BICpuElementwiseArithmetic<ArithmeticOperation::MAX>;
/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for minimum operation */
        using BICpuElementwiseMin = BICpuElementwiseArithmetic<ArithmeticOperation::MIN>;
/** Class to run @ref cpu::kernels::CpuArithmeticKernel except for squared difference operation */
        using BICpuElementwiseSquaredDiff = BICpuElementwiseArithmetic<ArithmeticOperation::SQUARED_DIFF>;

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be S32/F16/F32.
 * @note The function performs a division operation between two tensors (i.e., out[i] = in1[i] / in2[i])
 */
        class BICpuElementwiseDivision : public BICpuElementwiseBase {
        public:
            /** Initialise the kernel's inputs, dst and conversion policy.
             *
             * @param[in, out] src0 First tensor input info. Data types supported: S32/F16/F32.
             * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
             * @param[out]     dst  Output tensor info. Data types supported: Same as @p src0.
             */
            void configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref BICpuElementwiseDivision::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);
        };

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 * @note For an exponent that is a float, this function will only work with a positive base.
 */
        class BICpuElementwisePower : public BICpuElementwiseBase {
        public:
            /** Initialise the kernel's inputs, dst and conversion policy.
             *
             * @param[in, out] src0 First tensor input info. Data types supported: F16/F32.
             * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
             * @param[out]     dst  Output tensor info. Data types supported: Same as @p src0.
             */
            void configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref BICpuElementwisePower::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);
        };

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel.
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
        class BICpuElementwiseComparison : public BICpuElementwiseBase {
        public:
            /** Initialise the kernel's inputs, dst and conversion policy.
             *
             * @param[in, out] src0 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
             * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
             * @param[out]     dst  Output tensor info. Data types supported: U16/U32.
             * @param[in]      op   Comparison Operation to be performed.
             */
            void
            configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst, ComparisonOperation op);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref BICpuElementwiseComparison::configure()
             *
             * @return a status
             */
            static BIStatus
            validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst,
                     ComparisonOperation op);
        };

/** Basic function to run @ref cpu::kernels::CpuComparisonKernel
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
        template<ComparisonOperation op>
        class BICpuElementwiseComparisonStatic : public BICpuElementwiseBase {
        public:
            /** Initialise the kernel's inputs, dst and conversion policy.
             *
             * @param[in, out] src0 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
             * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
             * @param[out]     dst  Output tensor info. Data types supported: U16/U32.
             */
            void configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref BICpuElementwiseComparisonStatic::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst);
        };

/** Basic function to run equal comparison. */
        using NEEqual = BICpuElementwiseComparisonStatic<ComparisonOperation::Equal>;
/** Basic function to run not equal comparison. */
        using NENotEqual = BICpuElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
/** Basic function to run greater comparison. */
        using NEGreater = BICpuElementwiseComparisonStatic<ComparisonOperation::Greater>;
/** Basic function to run greater-equal comparison. */
        using NEGreaterEqual = BICpuElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
/** Basic function to run less comparison. */
        using NELess = BICpuElementwiseComparisonStatic<ComparisonOperation::Less>;
/** Basic function to run less-equal comparison. */
        using NELessEqual = BICpuElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
    } // namespace cpu
}