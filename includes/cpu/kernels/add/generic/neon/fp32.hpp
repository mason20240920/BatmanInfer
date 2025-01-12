//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_NEON_KERNELS_ADD_FP32_HPP
#define BATMANINFER_NEON_KERNELS_ADD_FP32_HPP

namespace BatmanInfer {
    class BIITensor;

    enum class BIConvertPolicy;

    class BIWindow;
    namespace cpu {
        void add_fp32_neon(
                const BIITensor *src0,
                const BIITensor *src1,
                BIITensor *dst, const
                BIConvertPolicy &policy,
                const BIWindow &window);

    }
}


#endif //BATMANINFER_NEON_KERNELS_ADD_FP32_HPP
