//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    namespace softmax_helpers {
/** Given a softmax axis, this function returns the permutation vector required to put the axis to the front
 *
 * @note This function assumes a tensor rank <= 4
 *
 * Axis selects the dimension on which softmax is performed.
 * E.g. For input of shape 4x5x6 and axis=1, softmax will be applied to 4x6=24 vectors of size 5.
 * Interally softmax kernels is always performed on the first dimension (front dimension), therefore permutation is
 * required to put the dimension specified by @p axis to the first dimension.
 *
 * @param[in] axis Axis on which to perform softmax. Supported: 1, 2, 3 (0 implies no permutation needed)
 *
 * @return the permutation vector
 */
        PermutationVector get_permutation_vector_from_softmax_axis(size_t axis);
    } // namespace softmax_helpers
}