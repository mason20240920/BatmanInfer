//
// Created by Mason on 2025/1/1.
//

#ifndef BATMANINFER_MATH_HPP
#define BATMANINFER_MATH_HPP

#include <data/core/bi_error.h>

namespace BatmanInfer {
    /**
     * @brief 计算 val / m 的向上取整商。
     * @tparam S
     * @tparam T
     * @param val
     * @param m
     * @return
     */
    template <typename S, typename T>
    constexpr auto DIV_CEIL(S val, T m) -> decltype((val + m - 1) / m) {
        return (val + m - 1) / m;
    }

    /**
     * @brief 计算大于或等于值的最小倍数。
     * @tparam S
     * @tparam T
     * @param value
     * @param divisor
     * @return
     */
    template <typename S, typename T>
    inline auto ceil_to_multiples(S value, T divisor) -> decltype(((value + divisor - 1) / divisor) * divisor) {
        BI_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
        return DIV_CEIL(value, divisor) * divisor;
    }

    /**
     * @brief 计算小于或等于值的最大数，该数是除数的倍数。
     * @tparam S
     * @tparam T
     * @param value
     * @param divisor
     * @return
     */
    template <typename S, typename T>
    inline auto floor_to_multiples(S value, T divisor) -> decltype((value / divisor) * divisor)
    {
        BI_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
        return (value / divisor) * divisor;
    }
}

#endif //BATMANINFER_MATH_HPP
