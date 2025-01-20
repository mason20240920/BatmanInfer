//
// Created by Mason on 2025/1/11.
//

#include "cpu/kernels/activation/generic/neon/impl.hpp"

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