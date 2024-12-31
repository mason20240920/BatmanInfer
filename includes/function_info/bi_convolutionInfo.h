//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_CONVOLUTIONINFO_H
#define BATMANINFER_CONVOLUTIONINFO_H

#include "data/core/core_types.hpp"
#include "data/core/bi_size_2D.h"
#include "function_info/bi_activationLayerInfo.h"

namespace BatmanInfer {

    struct BIConvolutionInfo
    {
        BIConvolutionInfo() = default;
        BIConvolutionInfo(const BIPadStrideInfo       &pad_stride_info,
                        unsigned int                   depth_multiplier,
                        const BIActivationLayerInfo   &act_info,
                        const Size2D                  &dilation)
            : pad_stride_info(pad_stride_info), depth_multiplier(depth_multiplier), act_info(act_info), dilation(dilation)
        {
        }
        BIPadStrideInfo pad_stride_info{}; /**< Convolution info (Pads, strides,...) */
        unsigned int    depth_multiplier{
            1}; /**< Multiplier to apply to input's depth to retrieve the output depth. Defaults to 1 */
        BIActivationLayerInfo act_info{};             /**< Fused activation to apply after convolution. */
        Size2D                dilation{Size2D(1, 1)}; /**< Dilation, in elements, across x and y. Defaults to (1, 1). */
    };

} // namespace BatmanInfer

#endif //BATMANINFER_CONVOLUTIONINFO_H
