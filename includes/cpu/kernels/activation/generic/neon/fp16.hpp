//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_KERNELS_ACTIVATION_FP16_HPP
#define BATMANINFER_KERNELS_ACTIVATION_FP16_HPP

namespace BatmanInfer {
    namespace cpu {

        void neon_fp16_activation(const BatmanInfer::BIITensor *src,
                                  BatmanInfer::BIITensor *dst,
                                  const BatmanInfer::BIActivationLayerInfo &act_info,
                                  const BatmanInfer::BIWindow &window);
    }
}

#endif //BATMANINFER_KERNELS_ACTIVATION_FP16_HPP
