//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_HALF_HPP
#define BATMANINFER_BI_HALF_HPP

// 这是一个预定义的宏，通常用于指代“裸机环境”（Bare Metal Environment）
#if (BARE_METAL)
#define HALF_ENABLE_CPP11_CMATH 0
#endif /* BARE_METAL */

// Set style to round to nearest
#define HALF_ROUND_STYLE        1
#define HALF_ROUND_TIES_TO_EVEN 1

#include "half/half.hpp"

#endif //BATMANINFER_BI_HALF_HPP
