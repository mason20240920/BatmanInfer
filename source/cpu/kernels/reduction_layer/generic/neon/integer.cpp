//
// Created by Mason on 2025/4/10.
//
#include <cpu/kernels/reduction_layer/generic/neon/integer.hpp>

#include <cpu/kernels/reduction_layer/generic/neon/impl.hpp>

namespace BatmanInfer::cpu {
    void reduce_RedOpX_reduceX_S32_4(const BIWindow &window,
                                     const BIITensor *input,
                                     BIITensor *output,
                                     const BIReductionOperation op) {
        return Reducer<RedOpX<int32_t, 4> >::reduceX(window, input, output, RedOpX<int32_t, 4>(), op);
    }

    void reduce_RedOpYZW_reduceY_S32_4(const BIWindow &window,
                                       const BIITensor *input,
                                       BIITensor *output,
                                       const BIReductionOperation op) {
        return Reducer<RedOpYZW<int32_t, 4> >::reduceY(window, input, output, RedOpYZW<int32_t, 4>(), op);
    }

    void reduce_RedOpYZW_reduceZ_S32_4(const BIWindow &window,
                                       const BIITensor *input,
                                       BIITensor *output,
                                       const BIReductionOperation op) {
        return Reducer<RedOpYZW<int32_t, 4> >::reduceZ(window, input, output, RedOpYZW<int32_t, 4>(), op);
    }

    void reduce_RedOpYZW_reduceW_S32_4(const BIWindow &window,
                                       const BIITensor *input,
                                       BIITensor *output,
                                       const BIReductionOperation op) {
        return Reducer<RedOpYZW<int32_t, 4> >::reduceW(window, input, output, RedOpYZW<int32_t, 4>(), op);
    }
}
