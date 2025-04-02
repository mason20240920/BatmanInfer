//
// Created by Mason on 2025/3/3.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/neon/bi_i_ne_simple_function_no_border.h>

namespace BatmanInfer {
    // Forward declarations
    class BIITensor;
    class BIITensorInfo;

    class BINEGather : public BIINESimpleFunctionNoBorder {
    public:
        /** Initialise the kernel's inputs and outputs
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |All            |All            |
         *
         * @param[in]  input   Source tensor. Supported tensor rank: up to 4. Data type supported: All
         * @param[in]  indices Indices tensor. Supported tensor rank: up to 3. Must be one of the following type: U32/S32. Each value must be in range [0, input.shape[@p axis]), otherwise the result will become unpredictable.
         *                     @note The "axis" must be in the range [0, input. Rank -1] when indices is a vector, and must be 1 when indices is a 2D or 3D tensor.
         * @param[out] output  Destination tensor. Data type supported: Same as @p input
         * @param[in]  axis    (Optional) The axis in @p input to gather @p indices from. Defaults to 0
         *
         */
        void configure(const BIITensor *input,
                       const BIITensor *indices,
                       BIITensor *output,
                       int axis = 0);

        /** Static function to check if given info will lead to a valid configuration
        *
        * Similar to @ref BINEGather::configure()
        *
        * @return a status
        */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *indices,
                                 const BIITensorInfo *output,
                                 int axis);

        /**
         * @brief 动态配置信息
         * @param indices
         * @param output
         */
        void dynamic_configure(const BIITensor *indices,
                               BIITensor *output) const;
    };
}
