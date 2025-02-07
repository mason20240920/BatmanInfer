//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <cpu/bi_i_cpu_operator.hpp>

#include <data/core/bi_types.hpp>

/** This file contains all available output stages for BIGEMMLowp.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref BINEGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */
namespace BatmanInfer {
    namespace cpu {

        /** Basic function to execute GEMMLowpQuantizeDown kernels.
        *
        *  This function calls the following kernels:
        *
        * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ScaleKernel
        * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
        * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel
        * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel
       */
        class BICpuGemmLowpOutputStage : public BIICpuOperator {
        public:
            /** Initialise the kernel's inputs, output
             *
             * Valid data layouts:
             * - All
             *
             * Valid data type configurations:
             * |src0           |src1          |dst           |
             * |:--------------|:-------------|:-------------|
             * |S32            |S32           |QASYMM8       |
             * |S32            |S32           |QASYMM8_SIGNED|
             * |S32            |S32           |QSYMM16       |
             *
             * @param[in]  src  Input tensor info. Data type supported: S32
             * @param[in]  bias Biases tensor info. Only shared biases supported and it can be a nullptr if the biases addition is not required.
             *                  Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
             * @param[out] dst  Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
             * @param[in]  info GEMMLowp output stage metadata.
             */
            void configure(BIITensorInfo *src, BIITensorInfo *bias, BIITensorInfo *dst,
                           const BIGEMMLowpOutputStageInfo &info);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to CpuGemmLowpOutputStage::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src,
                                     const BIITensorInfo *bias,
                                     const BIITensorInfo *dst,
                                     const BIGEMMLowpOutputStageInfo &info);

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;
        };
    } // namespace cpu
} // namespace BatmanInfer