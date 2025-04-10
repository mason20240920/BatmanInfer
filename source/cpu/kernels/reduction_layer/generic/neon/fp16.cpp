//
// Created by Mason on 2025/4/10.
//
#include <cpu/kernels/reduction_layer/generic/neon/fp16.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/impl.hpp>
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer::cpu {
    void reduce_RedOpX_reduceX_float16_8(const BIWindow &window,
                                         const BIITensor *input,
                                         BIITensor *output,
                                         const BIReductionOperation op) {
        return Reducer<RedOpX<float16_t, 8> >::reduceX(window, input, output, RedOpX<float16_t, 8>(), op);
    }

    void reduce_RedOpYZW_reduceY_float16_8(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op) {
        return Reducer<RedOpYZW<float16_t, 8> >::reduceY(window, input, output, RedOpYZW<float16_t, 8>(), op);
    }

    void reduce_RedOpYZW_reduceZ_float16_8(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op) {
        return Reducer<RedOpYZW<float16_t, 8> >::reduceZ(window, input, output, RedOpYZW<float16_t, 8>(), op);
    }

    void reduce_RedOpYZW_reduceW_float16_8(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op) {
        return Reducer<RedOpYZW<float16_t, 8> >::reduceW(window, input, output, RedOpYZW<float16_t, 8>(), op);
    }
}

#endif
