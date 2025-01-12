//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_BI_UTILS_TRAITS_TRAITS_HPP
#define BATMANINFER_BI_UTILS_TRAITS_TRAITS_HPP

#include "data/core/bi_types.hpp"

#include <type_traits>

namespace BatmanInfer {
    namespace utils {
        namespace traits {
            template<typename T>
            struct is_floating_point : public std::is_floating_point<T> {
            };

            template<>
            struct is_floating_point<half> : public std::true_type {
            };

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            template<>
            struct is_floating_point<__fp16> : public std::true_type {
            };
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/
        }
    }
}

#endif //BATMANINFER_BI_UTILS_TRAITS_TRAITS_HPP
