//
// Created by Mason on 2025/1/12.
//

#include "cpu/kernels/add/generic/neon/fp32.hpp"
#include "cpu/bi_cpu_types.hpp"
#include "cpu/kernels/add/generic/neon/impl.hpp"

namespace BatmanInfer {
    namespace cpu {
        void add_fp32_neon(
                const BIITensor *src0,
                const BIITensor *src1,
                BIITensor *dst, const
                BIConvertPolicy &policy,
                const BIWindow &window) {
            return add_same_neon<float>(src0, src1, dst, policy, window);
        }
    }
}