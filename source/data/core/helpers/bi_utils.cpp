//
// Created by Mason on 2025/1/8.
//

#include <data/core/utils/helpers/bi_utils.hpp>

namespace BatmanInfer {
    bool has_holes(const BIITensorInfo &info) {
        return has_holes(info, info.num_dimensions() - 1);
    }

    bool has_holes(const BIITensorInfo &info, size_t dimension) {
        const auto &shape = info.tensor_shape();
        const auto &strides = info.strides_in_bytes();
        size_t squashed_bytes = info.element_size();

        for (size_t dim = 0; dim <= dimension; ++dim) {
            if (strides[dim] != squashed_bytes) {
                return true;
            }
            squashed_bytes *= shape[dim];
        }
        return false;
    }
}