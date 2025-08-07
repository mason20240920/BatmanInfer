//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <data/core/bi_rounding.h>

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Interface for the kernel to perform multiplication between two tensors */
            class BICpuMulKernel : public BIICpuKernel<BICpuMulKernel> {
            public:
                BICpuMulKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuMulKernel);

                /** Initialise the kernel's input, dst and border mode.
                 *
                 * Valid configurations (Src1,Src2) -> Dst :
                 *
                 *                                                       Support: Broadcast? Scale=1/255?
                 *   - (U8,U8)                         -> U8, S16                 N          Y
                 *   - (U8,S16)                        -> S16                     N          Y
                 *   - (S16,U8)                        -> S16                     N          Y
                 *   - (S16,S16)                       -> S16                     N          Y
                 *   - (S32,S32)                       -> S32                     Y          N
                 *   - (F16,F16)                       -> F16                     N          Y
                 *   - (F32,F32)                       -> F32                     Y          Y
                 *   - (QASYMM8,QASYMM8)               -> QASYMM8                 Y          Y
                 *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED          Y          Y
                 *   - (QSYMM16,QSYMM16)               -> QSYMM16, S32            N          Y
                 *
                 * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
                 *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
                 *
                 * @param[in]  src1            First input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
                 * @param[in]  src2            Second input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
                 * @param[out] dst             Dst tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
                 * @param[in]  scale           Scale to apply after multiplication.
                 *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
                 *                             If both @p src1, @p src2 and @p dst are of datatype S32, scale cannot be 1/255
                 * @param[in]  overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
                 * @param[in]  rounding_policy Rounding policy.
                 */
                void configure(BIITensorInfo *src1,
                               BIITensorInfo *src2,
                               BIITensorInfo *dst,
                               float scale,
                               BIConvertPolicy overflow_policy,
                               BIRoundingPolicy rounding_policy);


                void dynamic_configure(BIITensorInfo *src);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to @ref CpuMulKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *src1,
                                         const BIITensorInfo *src2,
                                         const BIITensorInfo *dst,
                                         float scale,
                                         BIConvertPolicy overflow_policy,
                                         BIRoundingPolicy rounding_policy);

                // Inherited methods overridden
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

                /** Return minimum workload size of the relevant kernel
                 *
                 * @param[in] platform     The CPU platform used to create the context.
                 * @param[in] thread_count Number of threads in the execution.
                 *
                 * @return[out] mws Minimum workload size for requested configuration.
                 */
                size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

                /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
                  *
                  * @return The split dimension hint.
                  */
                size_t get_split_dimension_hint() const {
                    return _split_dimension;
                }

            private:
                /** Common signature for all the specialised multiplication functions with integer scaling factor
                 *
                 * @param[in]  src1   Src1 tensor object.
                 * @param[in]  src2   Src2 tensor object.
                 * @param[out] dst    Dst tensor object.
                 * @param[in]  window Region on which to execute the kernel
                 * @param[in]  scale  Integer scale factor.
                 */
                using MulFunctionInt =
                        void(const BIITensor *src1, const BIITensor *src2, BIITensor *dst, const BIWindow &window,
                             int scale);
                /** Common signature for all the specialised multiplication functions with float scaling factor
                 *
                 * @param[in]  src1   Src1 tensor object.
                 * @param[in]  src2   Src2 tensor object.
                 * @param[out] dst    Dst tensor object.
                 * @param[in]  window Region on which to execute the kernel
                 * @param[in]  scale  Float scale factor.
                 */
                using MulFunctionFloat =
                        void(const BIITensor *src1, const BIITensor *src2, BIITensor *dst, const BIWindow &window,
                             float scale);
                /** Common signature for all the specialised QASYMM8 multiplication functions with float scaling factor
                 *
                 * @param[in]  src1   Src1 tensor object.
                 * @param[in]  src2   Src2 tensor object.
                 * @param[out] dst    Dst tensor object.
                 * @param[in]  window Region on which to execute the kernel
                 * @param[in]  scale  Float scale factor.
                 *
                 */
                using MulFunctionQuantized =
                        void(const BIITensor *src1, const BIITensor *src2, BIITensor *dst, const BIWindow &window,
                             float scale);

                MulFunctionFloat *_func_float{nullptr};
                MulFunctionInt *_func_int{nullptr};
                MulFunctionQuantized *_func_quantized{nullptr};
                float _scale{0};
                int _scale_exponent{0};
                size_t _split_dimension{BIWindow::DimY};
            };

/** Interface for the complex pixelwise multiplication kernel. */
            class BICpuComplexMulKernel : public BIICpuKernel<BICpuComplexMulKernel> {
            public:
                BICpuComplexMulKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuComplexMulKernel);

                /** Initialise the kernel's src, dst and border mode.
                 *
                 * @param[in]  src1 An src tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
                 * @param[in]  src2 An src tensor. Data types supported: same as @p src1. Number of channels supported: same as @p src1.
                 * @param[out] dst  The dst tensor, Data types supported: same as @p src1.  Number of channels supported: same as @p src1.
                 */
                void configure(BIITensorInfo *src1, BIITensorInfo *src2, BIITensorInfo *dst);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to @ref CpuComplexMulKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src1, const BIITensorInfo *src2, const BIITensorInfo *dst);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;
            };
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer