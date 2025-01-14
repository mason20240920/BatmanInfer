//
// Created by holynova on 2025/1/15.
//

#pragma once

#include "support/bi_acl_requires.h"

#include <type_traits>

namespace BatmanInfer {

namespace helpers {

namespace bit_ops {

    /** Checks if the idx-th bit is set in an integral type
     *
     * @param[in] v   Integral input
     * @param[in] idx Index of the bit to check
     *
     * @return True if the idx-th bit is set else false
     */
    template <typename T, BI_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
    bool is_bit_set(T v, unsigned int idx)
    {
        return (v & 1 << idx) != 0;
    }

} // namespace bit_ops

} // namespace helpers

} // namespace BatmanInfer
