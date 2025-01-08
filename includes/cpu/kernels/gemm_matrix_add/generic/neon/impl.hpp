//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_KERNELS_GEMMMATRIXADD_IMPL_HPP
#define BATMANINFER_KERNELS_GEMMMATRIXADD_IMPL_HPP

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        void matrix_addition_f32(const BIITensor *src,
                                 BIITensor *dst,
                                 const BIWindow &window,
                                 float beta);
    }
}

#endif //BATMANINFER_KERNELS_GEMMMATRIXADD_IMPL_HPP
