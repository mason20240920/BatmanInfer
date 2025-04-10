//
// Created by Mason on 2025/4/10.
//
#include <cpu/kernels/reduction_layer/generic/neon/qasymm8.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/impl.hpp>

namespace BatmanInfer::cpu {
    void reduce_RedOpX_reduceX_qasymm8(const BIWindow &window,
                                       const BIITensor *input,
                                       BIITensor *output,
                                       const BIReductionOperation op) {
        return Reducer<RedOpX_quantized<uint8_t> >::reduceX(window, input, output, RedOpX_quantized<uint8_t>(), op);
    }

    void reduce_RedOpYZW_reduceY_qasymm8(const BIWindow &window,
                                         const BIITensor *input,
                                         BIITensor *output,
                                         const BIReductionOperation op) {
        return Reducer<RedOpYZW_quantized<uint8_t> >::reduceY(window, input, output, RedOpYZW_quantized<uint8_t>(), op);
    }

    void reduce_RedOpYZW_reduceZ_qasymm8(const BIWindow &window,
                                         const BIITensor *input,
                                         BIITensor *output,
                                         const BIReductionOperation op) {
        return Reducer<RedOpYZW_quantized<uint8_t> >::reduceZ(window, input, output, RedOpYZW_quantized<uint8_t>(), op);
    }

    void reduce_RedOpYZW_reduceW_qasymm8(const BIWindow &window,
                                         const BIITensor *input,
                                         BIITensor *output,
                                         const BIReductionOperation op) {
        return Reducer<RedOpYZW_quantized<uint8_t> >::reduceW(window, input, output, RedOpYZW_quantized<uint8_t>(), op);
    }
}
