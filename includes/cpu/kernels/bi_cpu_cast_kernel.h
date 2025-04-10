//
// Created by holynova on 25-4-8.
//

#pragma once

#include "data/core/common/bi_core_common_macros.hpp"
#include "cpu/bi_i_cpu_kernel.hpp"
#include "data/core/bi_i_tensor.hpp"

namespace BatmanInfer {

namespace cpu {

namespace kernels {

    /** 将给定的 tensor 转变为新的类型
     *
     * @note When casting between quantized types the scale and zeroPoint are ignored
     */
    class BICpuCastKernel : public BIICpuKernel<BICpuCastKernel> {
    private:
        using CastKernelPtr =
            std::add_pointer<void(const BIITensor *, BIITensor *, const ThreadInfo&, BIConvertPolicy, const BIWindow &)>::type;

    public:
        BICpuCastKernel() = default;
        BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuCastKernel);

        /** Set the src and dst of the kernel
         *
         * Valid conversions src -> dst :
         *
         *   - QASYMM8_SIGNED -> S16, S32, F32, F16
         *   - QASYMM8        -> U16, S16, S32, F32, F16
         *   - U8             -> U16, S16, S32, F32, F16
         *   - U16            -> U8, U32
         *   - S16            -> QASYMM8_SIGNED, U8, S32
         *   - F16            -> QASYMM8_SIGNED, QASYMM8, F32, S32, U8
         *   - S32            -> QASYMM8_SIGNED, QASYMM8, F16, F32, U8
         *   - S64            -> F32
         *   - F32            -> QASYMM8_SIGNED, QASYMM8, F16, S32, U8
         *
         * @param[in]  src    The src tensor to convert. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/U16/S16/S32/S64/F16/F32.
         * @param[out] dst    The dst tensor. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/U16/S16/U32/S32/S64/F16/F32.
         * @param[in]  policy Conversion policy.
         *
         * @note S64 is only supported in aarch64
         *
         */
        void configure(const BIITensorInfo *src, BIITensorInfo *dst, BIConvertPolicy policy);

        static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst, BIConvertPolicy policy);

        // Inherited methods overridden:
        void        run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;
        const char *name() const override;

        struct CastKernel
        {
            const char                          *name;
            const CastDataTypeISASelectorDataPtr is_selected;
            CastKernelPtr                        ukernel;
        };

        static const std::vector<CastKernel> &get_available_kernels();

    private:
        BIConvertPolicy _policy{BIConvertPolicy::SATURATE};
    };

} // namespace kernels

} // namespace cpu

} // namespace BatmanInfer
