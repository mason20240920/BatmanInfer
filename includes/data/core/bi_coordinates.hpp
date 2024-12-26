//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_COORDINATES_HPP
#define BATMANINFER_COORDINATES_HPP

#include <data/core/dimensions.hpp>

namespace BatmanInfer {
    /**
     * @brief 元素的坐标
     */
    class BICoordinates : public BIDimensions<int> {
    public:
        /**
         * @brief 初始化坐标
         * @tparam Ts
         * @param coords
         */
        template <typename... Ts>
        constexpr explicit BICoordinates(Ts... coords): BIDimensions{coords...}
        {

        }

        constexpr BICoordinates(const BICoordinates &) = default;

        BICoordinates &operator=(const BICoordinates &) = default;

        constexpr BICoordinates(BICoordinates &&) = default;

        BICoordinates &operator=(BICoordinates &&) = default;

        ~BICoordinates() = default;

    };
}

#endif //BATMANINFER_COORDINATES_HPP
