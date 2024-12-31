//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_BI_FULLYCONNECTEDLAYERINFO_H
#define BATMANINFER_BI_FULLYCONNECTEDLAYERINFO_H

#include "data/core/core_types.hpp"
#include "function_info/bi_activationLayerInfo.h"

namespace BatmanInfer {

    /** Fully connected layer info */
    struct FullyConnectedLayerInfo
    {
        /* Fused-activation parameters */
        BIActivationLayerInfo activation_info{}; /**<  Fused activation to apply after the matrix multiplication. */
        /* Information about weights */
        BIDataLayout weights_trained_layout{BIDataLayout::NCHW}; /**<  Layout that the weights have been trained with. */
        bool         transpose_weights{true};                    /**<  Transpose weights if true. */
        bool         are_weights_reshaped{false};                /**<  @deprecated Reshape the weights tensor if false. */
        bool         retain_internal_weights{false};             /**<  Retain internal reshaped weights. */
        bool         enable_fast_math{false};                    /**<  Enable fast math computation. */
        /* Other parameters */
        bool fp_mixed_precision{false}; /**<  Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy. */

        /** Sets the weights trained data layout
         *
         * @param[in] layout Data layout that the weights were trained with
         *
         * @return Updated object
         */
        FullyConnectedLayerInfo &set_weights_trained_layout(BIDataLayout layout)
        {
            weights_trained_layout = layout;
            return *this;
        }
        /** Sets the transpose weights flag
         *
         * @param[in] should_transpose_weights Boolean flag indicating if weights should be transposed
         *
         * @return Updated object
         */
        FullyConnectedLayerInfo &set_transpose_weights(bool should_transpose_weights)
        {
            transpose_weights = should_transpose_weights;
            return *this;
        }
    };

} // namespace BatmanInfer

#endif //BATMANINFER_BI_FULLYCONNECTEDLAYERINFO_H
