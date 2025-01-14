//
// Created by holynova on 2025/1/15.
//

#pragma once

namespace BatmanInfer {

namespace utils {

namespace detail {

    enum class enabler
    {
    };

} // namespace detail

/** Requirements as template */
#define BI_COMPUTE_REQUIRES_T(...) template <bool Cond = (__VA_ARGS__), typename std::enable_if<Cond, int>::type = 0>
/** Requirements as template argument */
#define BI_COMPUTE_REQUIRES_TA(...) typename = typename std::enable_if<(__VA_ARGS__), BatmanInfer::utils::detail::enabler>::type

} // namespace utils

} // namespace BatmanInfer
