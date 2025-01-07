//
// Created by holynova on 2025/1/3.
//
#include "data/core/bi_rounding.h"
#include "data/core/bi_error.h"
#include "support/bi_toolchain_support.hpp"

#include <cmath>

using namespace BatmanInfer;
using namespace std;

int BatmanInfer::round(float x, BIRoundingPolicy rounding_policy)
{
    using namespace std;
    int rounded = 0;
    switch (rounding_policy)
    {
    case BIRoundingPolicy::TO_ZERO:
        {
            rounded = static_cast<int>(x);
            break;
        }
    case BIRoundingPolicy::TO_NEAREST_UP:
        {
            rounded = static_cast<int>(support::cpp11::round(x));
            break;
        }
    case BIRoundingPolicy::TO_NEAREST_EVEN:
        {
#ifdef __aarch64__
            asm("fcvtns %x[res], %s[value]" : [res] "=r"(rounded) : [value] "w"(x));
#else  // __aarch64__
            BI_COMPUTE_ERROR("TO_NEAREST_EVEN rounding policy is not supported.");
#endif // __aarch64__
            break;
        }
    default:
        {
            BI_COMPUTE_ERROR("Unsupported rounding policy.");
            break;
        }
    }

    return rounded;
}




