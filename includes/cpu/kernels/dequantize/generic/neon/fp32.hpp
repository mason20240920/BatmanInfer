//
// Created by Mason on 2025/2/8.
//

#pragma once

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void fp32_run_dequantization_core(const BIITensor *input, BIITensor *output, const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer