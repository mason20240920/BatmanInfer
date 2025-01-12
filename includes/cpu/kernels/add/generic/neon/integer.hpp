//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_NEON_KERNELS_ADD_INTEGER_HPP
#define BATMANINFER_NEON_KERNELS_ADD_INTEGER_HPP

namespace BatmanInfer {
    class BIITensor;

    enum class BIConvertPolicy;

    class BIWindow;

    namespace cpu {
        void add_u8_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window);

        void add_s16_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window);

        void add_s32_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window);
    }
}

#endif //BATMANINFER_NEON_KERNELS_ADD_INTEGER_HPP
