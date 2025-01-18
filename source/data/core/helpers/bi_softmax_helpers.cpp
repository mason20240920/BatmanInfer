//
// Created by Mason on 2025/1/18.
//

#include <data/core/helpers/bi_softmax_helpers.hpp>

namespace BatmanInfer {
    namespace softmax_helpers {
        PermutationVector get_permutation_vector_from_softmax_axis(size_t axis) {
            switch (axis) {
                case 1:
                    return PermutationVector(1U, 0U, 2U, 3U);
                case 2:
                    return PermutationVector(2U, 1U, 0U, 3U);
                case 3:
                    return PermutationVector(3U, 1U, 2U, 0U);
                default:
                    BI_COMPUTE_ERROR("Axis not supported");
            }
        }
    } // namespace softmax_helpers
}