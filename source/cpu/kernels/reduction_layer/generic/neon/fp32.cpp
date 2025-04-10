//
// Created by Mason on 2025/4/10.
//
#include <cpu/kernels/reduction_layer/generic/neon/fp32.hpp>

namespace BatmanInfer::cpu {
    void reduce_RedOpYZW_complex_reduceZ_float32_4_2_SUM(const BIWindow &window,
                                                         const BIITensor *input,
                                                         BIITensor *output,
                                                         const BIReductionOperation op) {
        Reducer<RedOpYZW_complex<float, 4, 2, BIReductionOperation::SUM> >::reduceZ(
            window, input, output, RedOpYZW_complex<float, 4, 2, BIReductionOperation::SUM>(), op);
    }

    void reduce_RedOpX_reduceX_float32_4(const BIWindow &window,
                                         const BIITensor *input,
                                         BIITensor *output,
                                         const BIReductionOperation op) {
        return Reducer<RedOpX<float, 4> >::reduceX(window, input, output, RedOpX<float, 4>(), op);
    }

    void reduce_RedOpYZW_reduceY_float32_4(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op) {
        return Reducer<RedOpYZW<float, 4> >::reduceY(window, input, output, RedOpYZW<float, 4>(), op);
    }

    void reduce_RedOpYZW_reduceZ_float32_4(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op) {
        return Reducer<RedOpYZW<float, 4> >::reduceZ(window, input, output, RedOpYZW<float, 4>(), op);
    }

    void reduce_RedOpYZW_reduceW_float32_4(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op) {
        return Reducer<RedOpYZW<float, 4> >::reduceW(window, input, output, RedOpYZW<float, 4>(), op);
    }
}
