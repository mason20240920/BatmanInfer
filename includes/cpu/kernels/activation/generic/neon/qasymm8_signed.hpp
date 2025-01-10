//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_KERNELS_ACTIVATION_QASYMM8_SIGNED_HPP
#define BATMANINFER_KERNELS_ACTIVATION_QASYMM8_SIGNED_HPP

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_window.hpp>
#include <function_info/bi_activationLayerInfo.h>

#include <data/core/neon/bi_ne_asymm.hpp>
#include <data/core/neon/bi_neon_math.hpp>

#include <neon/neon_defines.h>
#include <cmath>

namespace BatmanInfer {
    namespace cpu {
        void neon_qasymm8_signed_activation(const BIITensor *src,
                                            BIITensor *dst,
                                            const BIActivationLayerInfo &act_info,
                                            const BIWindow &window);
    }
}

#endif //BATMANINFER_KERNELS_ACTIVATION_QASYMM8_SIGNED_HPP
