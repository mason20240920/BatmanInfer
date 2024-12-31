//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_STRIDES_HPP
#define BATMANINFER_STRIDES_HPP

#include <data/core/dimensions.hpp>

#include <cstdint>

namespace BatmanInfer {
    /**
     * @brief 一个元素的字节步数
     */
    class BIStrides : public BIDimensions<uint32_t>
    {
    public:
        /**
         * @brief 初始化步数
         * @tparam Ts
         * @param strides
         */
        template <typename... Ts>
        constexpr BIStrides(Ts... strides) : BIDimensions{strides...}
        {

        }

        /** Allow instances of this class to be copy constructed */
        constexpr BIStrides(const BIStrides &) = default;
        /** Allow instances of this class to be copied */
        BIStrides &operator=(const BIStrides &) = default;
        /** Allow instances of this class to be move constructed */
        constexpr BIStrides(BIStrides &&) = default;
        /** Allow instances of this class to be moved */
        BIStrides &operator=(BIStrides &&) = default;
        /** Default destructor */
        ~BIStrides() = default;
    };
}


#endif //BATMANINFER_STRIDES_HPP
