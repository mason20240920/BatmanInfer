//
// Created by Mason on 2025/1/12.
//

#include <cpu/kernels/add/generic/neon/integer.hpp>
#include <cpu/bi_cpu_types.hpp>
#include <cpu/kernels/add/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        void add_u8_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window) {
            return add_same_neon<uint8_t>(src0, src1, dst, policy, window);
        }

        void add_s16_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window) {
            return add_same_neon<int16_t>(src0, src1, dst, policy, window);
        }

        void add_s32_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window) {
            return add_same_neon<int32_t>(src0, src1, dst, policy, window);
        }
    }
}