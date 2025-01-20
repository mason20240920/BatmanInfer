//
// Created by Mason on 2025/1/12.
//

#include "cpu/kernels/add/generic/neon/qasymm8.hpp"
#include "data/core/bi_i_tensor.hpp"
#include "data/core/bi_types.hpp"
#include "cpu/kernels/add/generic/neon/impl.hpp"

namespace BatmanInfer {
    namespace cpu {
        void add_qasymm8_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window) {
            add_sub_qasymm8_neon(src0, src1, dst, policy, window, true /*is_addition*/);
        }
    } // namespace cpu
} // namespace arm_compute