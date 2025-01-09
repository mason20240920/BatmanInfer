//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_CPU_TYPES_HPP
#define BATMANINFER_BI_CPU_TYPES_HPP

namespace BatmanInfer {
    /* Type definitions compatible with arm_neon.h and arm_sve.h */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    typedef __fp16 float16_t;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    typedef float float32_t;
}

#endif //BATMANINFER_BI_CPU_TYPES_HPP
