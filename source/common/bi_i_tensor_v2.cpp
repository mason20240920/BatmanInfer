//
// Created by Mason on 2025/1/10.
//

#include <common/bi_i_tensor_v2.hpp>

#include <data/core/bi_tensor_info.hpp>

#include <common/utils/legacy_support.hpp>

namespace BatmanInfer {
    size_t BIITensorV2::get_size() const {
        return tensor()->info()->total_size();
    }

    BclTensorDescriptor BIITensorV2::get_descriptor() const {
        return detail::convert_to_descriptor(*tensor()->info());
    }
}