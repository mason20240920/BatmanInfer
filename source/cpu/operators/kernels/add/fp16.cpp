//
// Created by Mason on 2025/1/12.
//

#include <cpu/kernels/add/generic/neon/fp16.hpp>
#include <cpu/bi_cpu_types.hpp>
#include <cpu/kernels/add/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        void add_fp16_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window) {
            return add_same_neon<float16_t>(src0, src1, dst, policy, window);
        }
    }
}