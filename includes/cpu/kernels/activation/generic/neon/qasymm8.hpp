//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_KERNELS_ACTIVATION_QASYMM8_HPP
#define BATMANINFER_KERNELS_ACTIVATION_QASYMM8_HPP

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_qasymm8_activation(const BIITensor *src,
                                     BIITensor *dst,
                                     const BIActivationLayerInfo &act_info,
                                     const BIWindow &window);
    }
}

#endif //BATMANINFER_KERNELS_ACTIVATION_QASYMM8_HPP
