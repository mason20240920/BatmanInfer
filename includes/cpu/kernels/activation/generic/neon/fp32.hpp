//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_KERNELS_ACTIVATION_FP32_HPP
#define BATMANINFER_KERNELS_ACTIVATION_FP32_HPP

#include <cpu/kernels/activation/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace {
            constexpr ActFpImplParams Fp32Params = {static_cast<float>(1e-24), 4};
        }

        void
        neon_fp32_activation(const BatmanInfer::BIITensor *src,
                             BatmanInfer::BIITensor *dst, const
                             BatmanInfer::BIActivationLayerInfo &act_info,
                             const BatmanInfer::BIWindow &window) {
            fp_neon_activation_impl<float, Fp32Params>(src, dst, act_info, window);
        }
    } // namespace
}

#endif //BATMANINFER_KERNELS_ACTIVATION_FP32_HPP
