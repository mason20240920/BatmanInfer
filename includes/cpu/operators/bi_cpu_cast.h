//
// Created by holynova on 25-4-9.
//

#pragma once

#include "cpu/bi_i_cpu_operator.hpp"
#include "data/core/bi_i_tensor_info.hpp"


namespace BatmanInfer {

namespace cpu {

    /** Basic function to run @ref kernels::CpuCastKernel */
    class BICpuCast : public BIICpuOperator
    {
    public:
        /** Configure operator for a given list of arguments
         *
         * 输入类型必须与输出类型不同
         *
         * 允许的类型转换设置:
         * |src            |dst                                             |
         * |:--------------|:-----------------------------------------------|
         * |QASYMM8_SIGNED | S16, S32, F32, F16                             |
         * |QASYMM8        | U16, S16, S32, F32, F16                        |
         * |U8             | U16, S16, S32, F32, F16                        |
         * |U16            | U8, U32                                        |
         * |S16            | QASYMM8_SIGNED, U8, S32                        |
         * |F16            | QASYMM8_SIGNED, QASYMM8, F32, S32, U8          |
         * |S32            | QASYMM8_SIGNED, QASYMM8, F16, F32, U8          |
         * |F32            | QASYMM8_SIGNED, QASYMM8, F16, S32, U8|
         * |S64            | F32                                            |
         *
         * @param[in]  src    要转换的源向量. 允许的数据类型: U8/S8/U16/S16/U32/S32/S64/F16/F32.
         * @param[out] dst    目标向量. 允许的数据类型: U8/S8/U16/S16/U32/S32/F16/F32.
         * @param[in]  policy Conversion policy.
         *
         *
         */
        void configure(const BIITensorInfo *src, BIITensorInfo *dst, BIConvertPolicy policy);

        /** 静态函数，检测输入输出是否有效
         *
         * Similar to @ref CpuCast::configure()
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst, BIConvertPolicy policy);
    };

} // namespace cpu

} // namespace BatmanInfer

