//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/neon/bi_i_ne_simple_function_no_border.h"

namespace BatmanInfer {

    // Forward declarations
    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run NEChannelShuffleLayerKernel
 *
 * @note The function performs a channel shuffle operation on the input tensor. Given NCHW tensor with group G, it will
 * first divide the channels into G groups, C = (G * C'), and perform a transpose of the channel, which gives C = (C' * G).
 * for more details see: https://arxiv.org/pdf/1707.01083.pdf
 */
    class BINEChannelShuffleLayer : public BIINESimpleFunctionNoBorder
    {
    public:
        /** Initialize the function
         *
         * Valid data layouts:
         * - NCHW
         * - NHWC
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |All            |All            |
         *
         * @param[in]  input      Input tensor. Data types supported: All
         * @param[out] output     Output tensor. Data type supported: Same as @p input
         * @param[in]  num_groups Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
         */
        void configure(const BIITensor *input, BIITensor *output, unsigned int num_groups);
        /** Static function to check if given info will lead to a valid configuration of @ref NEChannelShuffleLayer
         *
         * @param[in]  input      Input tensor. Data types supported: All
         * @param[out] output     Output tensor. Data type supported: Same as @p input
         * @param[in]  num_groups Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output, unsigned int num_groups);
    };

} // namespace BatmanInfer
