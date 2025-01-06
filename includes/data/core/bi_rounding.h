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

    /** Return a rounded value of x. Rounding is done according to the rounding_policy.
     *
     * @param[in] x               Float value to be rounded.
     * @param[in] rounding_policy Policy determining how rounding is done.
     *
     * @return Rounded value of the argument x.
     */
    int round(float x, BIRoundingPolicy rounding_policy);

} // namespace BatmanInfer

#endif //BATMANINFER_BI_ROUNDING_H
