//
// Created by Mason on 2025/4/10.
//

#pragma once
#include <data/core/bi_types.hpp>

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
namespace BatmanInfer {
    class BIWindow;
    class BIITensor;

    namespace cpu {
        void reduce_RedOpX_reduceX_float16_8(const BIWindow &window,
                                             const BIITensor *input,
                                             BIITensor *output,
                                             const BIReductionOperation op);

        void reduce_RedOpYZW_reduceY_float16_8(const BIWindow &window,
                                               const BIITensor *input,
                                               BIITensor *output,
                                               const BIReductionOperation op);

        void reduce_RedOpYZW_reduceZ_float16_8(const BIWindow &window,
                                               const BIITensor *input,
                                               BIITensor *output,
                                               const BIReductionOperation op);

        void reduce_RedOpYZW_reduceW_float16_8(const BIWindow &window,
                                               const BIITensor *input,
                                               BIITensor *output,
                                               const BIReductionOperation op);
    }
}
#endif
