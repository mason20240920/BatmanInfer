//
// Created by Mason on 2025/1/14.
//

#pragma once

namespace BatmanGemm {

    struct PerformanceParameters {
        float kernel_macs_cycle;
        float prepare_bytes_cycle = 0.0f;
        float merge_bytes_cycle = 0.0f;

        PerformanceParameters(float k) : kernel_macs_cycle(k) {}

        PerformanceParameters(float k, float p, float m) : kernel_macs_cycle(k), prepare_bytes_cycle(p),
                                                           merge_bytes_cycle(m) {}
    };

} // namespace arm_gemm
