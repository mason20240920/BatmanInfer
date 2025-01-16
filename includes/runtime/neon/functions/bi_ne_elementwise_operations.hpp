//
// Created by Mason on 2025/1/16.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <function_info/bi_activationLayerInfo.h>
#include <runtime/bi_i_function.hpp>
#include <runtime/neon/bi_i_ne_operator.hpp>

namespace BatmanInfer {
    class BIITensor;

    /** Basic function to run cpu::kernels::CpuArithmeticKernel for max
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a max operation between two tensors.
 */
    class BINEElementwiseMax : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwiseMax();

        /** Default Destructor */
        ~BINEElementwiseMax();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseMax(const BINEElementwiseMax &) = delete;

        /** Default move constructor */
        BINEElementwiseMax(BINEElementwiseMax &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseMax &operator=(const BINEElementwiseMax &) = delete;

        /** Default move assignment operator */
        BINEElementwiseMax &operator=(BINEElementwiseMax &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |dst            |
         * |:--------------|:--------------|:--------------|
         * |QASYMM8        |QASYMM8        |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
         * |S32            |S32            |S32            |
         * |S16            |S16            |S16            |
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in, out] input1   First tensor input. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(BIITensor *input1,
                       BIITensor *input2,
                       BIITensor *output,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuArithmeticKernel for max
         *
         * @param[in] input1   First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run cpu::kernels::CpuArithmeticKernel for min
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a min operation between two tensors.
 */
    class BINEElementwiseMin : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwiseMin();

        /** Default Destructor */
        ~BINEElementwiseMin();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseMin(const BINEElementwiseMin &) = delete;

        /** Default move constructor */
        BINEElementwiseMin(BINEElementwiseMin &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseMin &operator=(const BINEElementwiseMin &) = delete;

        /** Default move assignment operator */
        BINEElementwiseMin &operator=(BINEElementwiseMin &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |dst            |
         * |:--------------|:--------------|:--------------|
         * |QASYMM8        |QASYMM8        |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
         * |S32            |S32            |S32            |
         * |S16            |S16            |S16            |
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in, out] input1   First tensor input. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(BIITensor *input1,
                       BIITensor *input2,
                       BIITensor *output,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuArithmeticKernel for min
         *
         * @param[in] input1   First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run cpu::kernels::CpuArithmeticKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
    class BINEElementwiseSquaredDiff : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwiseSquaredDiff();

        /** Default Destructor */
        ~BINEElementwiseSquaredDiff();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseSquaredDiff(const BINEElementwiseSquaredDiff &) = delete;

        /** Default move constructor */
        BINEElementwiseSquaredDiff(BINEElementwiseSquaredDiff &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseSquaredDiff &operator=(const BINEElementwiseSquaredDiff &) = delete;

        /** Default move assignment operator */
        BINEElementwiseSquaredDiff &operator=(BINEElementwiseSquaredDiff &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |dst            |
         * |:--------------|:--------------|:--------------|
         * |QASYMM8        |QASYMM8        |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
         * |S32            |S32            |S32            |
         * |S16            |S16            |S16            |
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in, out] input1   First tensor input. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(BIITensor *input1,
                       BIITensor *input2,
                       BIITensor *output,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuArithmeticKernel for squared difference
         *
         * @param[in] input1   First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run cpu::kernels::CpuArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = in1[i] / in2[i])
 */
    class BINEElementwiseDivision : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwiseDivision();

        /** Default Destructor */
        ~BINEElementwiseDivision();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseDivision(const BINEElementwiseDivision &) = delete;

        /** Default move constructor */
        BINEElementwiseDivision(BINEElementwiseDivision &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseDivision &operator=(const BINEElementwiseDivision &) = delete;

        /** Default move assignment operator */
        BINEElementwiseDivision &operator=(BINEElementwiseDivision &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |dst            |
         * |:--------------|:--------------|:--------------|
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in, out] input1   First tensor input. Data types supported: F16/F32.
         * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(BIITensor *input1,
                       BIITensor *input2,
                       BIITensor *output,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuArithmeticKernel for division
         *
         * @param[in] input1   First tensor input info. Data types supported: F16/F32.
         * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run cpu::kernels::CpuArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs a elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 * @note For an exponent that is a float, this function will only work with a positive base.
 */
    class BINEElementwisePower : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwisePower();

        /** Default Destructor */
        ~BINEElementwisePower();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwisePower(const BINEElementwisePower &) = delete;

        /** Default move constructor */
        BINEElementwisePower(BINEElementwisePower &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwisePower &operator=(const BINEElementwisePower &) = delete;

        /** Default move assignment operator */
        BINEElementwisePower &operator=(BINEElementwisePower &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |dst            |
         * |:--------------|:--------------|:--------------|
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in, out] input1   First tensor input. Data types supported: F16/F32.
         * @param[in, out] input2   Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(BIITensor *input1,
                       BIITensor *input2,
                       BIITensor *output,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuArithmeticKernel for power
         *
         * @param[in] input1   First tensor input info. Data types supported: F16/F32.
         * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run cpu::kernels::CpuComparisonKernel.
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
    class BINEElementwiseComparison : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwiseComparison();

        /** Default Destructor */
        ~BINEElementwiseComparison();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseComparison(const BINEElementwiseComparison &) = delete;

        /** Default move constructor */
        BINEElementwiseComparison(BINEElementwiseComparison &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseComparison &operator=(const BINEElementwiseComparison &) = delete;

        /** Default move assignment operator */
        BINEElementwiseComparison &operator=(BINEElementwiseComparison &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |dst   |
         * |:--------------|:--------------|:-----|
         * |QASYMM8        |QASYMM8        |U8    |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |U8    |
         * |S32            |S32            |U8    |
         * |U8             |U8             |U8    |
         * |S16            |S16            |U8    |
         * |F16            |F16            |U8    |
         * |F32            |F32            |U8    |
         *
         * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output Output tensor. Data types supported: U8.
         * @param[in]      op     Comparison Operation to be performed.
         */
        void configure(BIITensor *input1, BIITensor *input2, BIITensor *output, ComparisonOperation op);

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuComparisonKernel
         *
         * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output Output tensor info. Data types supported: U8.
         * @param[in] op     Comparison Operation to be performed.
         *
         * @return a status
         */
        static BIStatus
        validate(const BIITensorInfo *input1, const BIITensorInfo *input2, const BIITensorInfo *output,
                 ComparisonOperation op);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run cpu::kernels::CpuComparisonKernel
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
 * @note The function performs a comparison operation between two tensors.
 */
    template<ComparisonOperation op>
    class BINEElementwiseComparisonStatic : public BIIFunction {
    public:
        /** Default Constructor */
        BINEElementwiseComparisonStatic();

        /** Default Destructor */
        ~BINEElementwiseComparisonStatic();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseComparisonStatic(const BINEElementwiseComparisonStatic &) = delete;

        /** Default move constructor */
        BINEElementwiseComparisonStatic(BINEElementwiseComparisonStatic &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEElementwiseComparisonStatic &operator=(const BINEElementwiseComparisonStatic &) = delete;

        /** Default move assignment operator */
        BINEElementwiseComparisonStatic &operator=(BINEElementwiseComparisonStatic &&);

        /** Initialise the kernel's inputs, output and conversion policy.
         *
         * @param[in, out] input1 First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in, out] input2 Second tensor input. Data types supported: Same as @p input1.
         * @param[out]     output Output tensor. Data types supported: U16/U32.
         */
        void configure(BIITensor *input1, BIITensor *input2, BIITensor *output);

        /** Static function to check if given info will lead to a valid configuration of cpu::kernels::CpuComparisonKernel
         *
         * @param[in] input1 First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
         * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
         * @param[in] output Output tensor info. Data types supported: U16/U32.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1, const BIITensorInfo *input2, const BIITensorInfo *output);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

/** Basic function to run equal comparison. */
    using NEEqual = BINEElementwiseComparisonStatic<ComparisonOperation::Equal>;
/** Basic function to run not equal comparison. */
    using NENotEqual = BINEElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
/** Basic function to run greater comparison. */
    using NEGreater = BINEElementwiseComparisonStatic<ComparisonOperation::Greater>;
/** Basic function to run greater-equal comparison. */
    using NEGreaterEqual = BINEElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
/** Basic function to run less comparison. */
    using NELess = BINEElementwiseComparisonStatic<ComparisonOperation::Less>;
/** Basic function to run less-equal comparison. */
    using NELessEqual = BINEElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
}