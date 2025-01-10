//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_CPU_KERNELS_ACTIVATION_LUT_HPP
#define BATMANINFER_CPU_KERNELS_ACTIVATION_LUT_HPP

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;

    class BIActivationLayerInfo;
    namespace cpu {
#ifdef __aarch64__

        void neon_q8_activation_lut(const BIITensor *src,
                                    BIITensor *dst,
                                    const BIActivationLayerInfo &act_info,
                                    const BIWindow &window);

#endif
    }
}

#endif //BATMANINFER_CPU_KERNELS_ACTIVATION_LUT_HPP
