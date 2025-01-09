//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_UTILS_MATH_HPP
#define BATMANINFER_BI_UTILS_MATH_HPP

#include <array>

namespace BatmanInfer {
    /** erf(x) 计算的查找表 */
    template<typename T>
    struct ErfLutEntry {
        /** erf(x) */
        T value;

        /** 2 / sqrt(pi) * e^(-x^2) */
        T scale;
    };

    /** The lookup table for FP32 erf(x) calculation. */
    extern const std::array<ErfLutEntry<float>, 513> erf_f32_lut;
}

#endif //BATMANINFER_BI_UTILS_MATH_HPP
