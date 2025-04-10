//
// Created by Mason on 2025/4/10.
//

#pragma once
#include <data/core/bi_types.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/impl.hpp>


namespace BatmanInfer {
    namespace cpu {
        void reduce_RedOpYZW_complex_reduceZ_float32_4_2_SUM(const BIWindow &window,
                                                             const BIITensor *input,
                                                             BIITensor *output,
                                                             const BIReductionOperation op);

        void reduce_RedOpX_reduceX_float32_4(const BIWindow &window,
                                             const BIITensor *input,
                                             BIITensor *output,
                                             const BIReductionOperation op);

        void reduce_RedOpYZW_reduceY_float32_4(const BIWindow &window,
                                               const BIITensor *input,
                                               BIITensor *output,
                                               const BIReductionOperation op);

        void reduce_RedOpYZW_reduceZ_float32_4(const BIWindow &window,
                                               const BIITensor *input,
                                               BIITensor *output,
                                               const BIReductionOperation op);

        void reduce_RedOpYZW_reduceW_float32_4(const BIWindow &window,
                                               const BIITensor *input,
                                               BIITensor *output,
                                               const BIReductionOperation op);
    }
}
