//
// Created by Mason on 2025/1/11.
//

#include <cpu/kernels/activation/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace {
            constexpr ActFpImplParams Fp16Params = {static_cast<float16_t>(1e-7), 8};
        } // namespace

        void neon_fp16_activation(const BatmanInfer::BIITensor *src,
                                  BatmanInfer::BIITensor *dst,
                                  const BatmanInfer::BIActivationLayerInfo &act_info,
                                  const BatmanInfer::BIWindow &window) {
            fp_neon_activation_impl<float16_t, Fp16Params>(src, dst, act_info, window);
        }
    }
}