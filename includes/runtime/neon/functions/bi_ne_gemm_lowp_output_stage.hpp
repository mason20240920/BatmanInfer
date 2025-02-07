//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>

/** This file contains all available output stages for GEMMLowp.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace BatmanInfer {
    class BIITensor;

    class BIITensorInfo;

    class BINEGEMMLowpOutputStage : public BIIFunction {
    public:
        /** Constructor */
        BINEGEMMLowpOutputStage();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEGEMMLowpOutputStage(const BINEGEMMLowpOutputStage &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEGEMMLowpOutputStage &operator=(const BINEGEMMLowpOutputStage &) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEGEMMLowpOutputStage(BINEGEMMLowpOutputStage &&) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEGEMMLowpOutputStage &operator=(BINEGEMMLowpOutputStage &&) = delete;

        /** Default destructor */
        ~BINEGEMMLowpOutputStage();

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
         * @param[in]  input  Input tensor. Data type supported: S32
         * @param[in]  bias   Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
         *                    Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
         * @param[out] output Output tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
         * @param[in]  info   GEMMLowp output stage metadata.
         */
        void configure(const BIITensor *input, const BIITensor *bias, BIITensor *output,
                       const BIGEMMLowpOutputStageInfo &info);

        /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpOutputStage
         *
         * @param[in] input  Input tensor info. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
         * @param[in] bias   Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
         *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
         * @param[in] output Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
         * @param[in] info   GEMMLowp output stage metadata.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *bias,
                                 const BIITensorInfo *output,
                                 const BIGEMMLowpOutputStageInfo &info);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer