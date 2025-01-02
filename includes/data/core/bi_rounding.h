//
// Created by holynova on 2025/1/2.
//

#ifndef BATMANINFER_BI_ROUNDING_H
#define BATMANINFER_BI_ROUNDING_H

namespace BatmanInfer {

    /** Rounding method */
    enum class BIRoundingPolicy
    {
        TO_ZERO,         /**< Truncates the least significant values that are lost in operations. */
        TO_NEAREST_UP,   /**< Rounds to nearest value; half rounds away from zero */
        TO_NEAREST_EVEN, /**< Rounds to nearest value; half rounds to nearest even */
    };

} // namespace BatmanInfer

#endif //BATMANINFER_BI_ROUNDING_H
