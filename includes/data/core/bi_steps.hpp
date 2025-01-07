//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_STEPS_HPP
#define BATMANINFER_BI_STEPS_HPP

#include <data/core/dimensions.hpp>
#include <data/core/bi_error.h>
#include <data/core/bi_types.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

namespace BatmanInfer {
    /** 类用于描述每个维度中的元素数量。类似于
     * @ref * Strides，但不是以字节为单位，
     * 而是以元素数量为单位。 */
    class BISteps : public BIDimensions<uint32_t> {
    public:
        /** Constructor to initialize the steps.
         *
         * @param[in] steps Values to initialize the steps.
         */
        template<typename... Ts>
        BISteps(Ts... steps) : BIDimensions{steps...} {
            // Initialize empty dimensions to 1
            std::fill(_id.begin() + _num_dimensions, _id.end(), 1);
        }

        /** Allow instances of this class to be copy constructed */
        constexpr BISteps(const BISteps &) = default;

        /** Allow instances of this class to be copied */
        BISteps &operator=(const BISteps &) = default;

        /** Allow instances of this class to be move constructed */
        constexpr BISteps(BISteps &&) = default;

        /** Allow instances of this class to be moved */
        BISteps &operator=(BISteps &&) = default;

        /** Default destructor */
        ~BISteps() = default;
    };
}

#endif //BATMANINFER_BI_STEPS_HPP
