//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_EXPERIMENTAL_TYPES_HPP
#define BATMANINFER_EXPERIMENTAL_TYPES_HPP

#include <data/core/bi_i_tensor_pack.hpp>
#include <data/bi_tensor_shape.hpp>

#include <vector>

namespace BatmanInfer {
    /**
     * @brief 内存类型
     */
    enum BITensorType : int32_t
    {
        ACL_UNKNOWN = -1,
        ACL_SRC_DST = 0,

        // Src
        ACL_SRC     = 0,
        ACL_SRC_0   = 0,
        ACL_SRC_1   = 1,
        ACL_SRC_2   = 2,
        ACL_SRC_3   = 3,
        ACL_SRC_4   = 4,
        ACL_SRC_5   = 5,
        ACL_SRC_6   = 6,
        ACL_SRC_END = 6,

        // Dst
        ACL_DST     = 30,
        ACL_DST_0   = 30,
        ACL_DST_1   = 31,
        ACL_DST_2   = 32,
        ACL_DST_END = 32,

        // Aux
        ACL_INT     = 50,
        ACL_INT_0   = 50,
        ACL_INT_1   = 51,
        ACL_INT_2   = 52,
        ACL_INT_3   = 53,
        ACL_INT_4   = 54,
        ACL_SRC_VEC = 256,
        ACL_DST_VEC = 512,
        ACL_INT_VEC = 1024,

        // Aliasing Types
        // Conv etc
        ACL_BIAS = ACL_SRC_2,

        // Gemm
        ACL_VEC_ROW_SUM = ACL_SRC_3,
        ACL_VEC_COL_SUM = ACL_SRC_4,
        ACL_SHIFTS      = ACL_SRC_5,
        ACL_MULTIPLIERS = ACL_SRC_6,
    };

    namespace experimental {
        enum class MemoryLifetime
        {
            Temporary  = 0,
            Persistent = 1,
            Prepare    = 2,
        };

        struct BIMemoryInfo {
            BIMemoryInfo() = default;

            explicit BIMemoryInfo(int slot, size_t size, size_t alignment = 0) noexcept : slot(slot), size(size), alignment(alignment)
            {

            }

            BIMemoryInfo(int slot, MemoryLifetime lifetime, size_t size, size_t alignment = 0) noexcept
                    : slot(slot), lifetime(lifetime), size(size), alignment(alignment)
            {
            }

            bool merge(int _slot, size_t new_size, size_t new_alignment = 0) noexcept {
                if (_slot != this->slot)
                    return false;

                size = std::max(size, new_size);
                alignment = std::max(alignment, new_alignment);

                return true;
            }

            int slot{ACL_UNKNOWN};
            MemoryLifetime lifetime{MemoryLifetime::Temporary};
            size_t size{0};
            size_t alignment{64};
        };

        using BIMemoryRequirements = std::vector<BIMemoryInfo>;
    }
}

#endif //BATMANINFER_EXPERIMENTAL_TYPES_HPP
