//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <data/core/utils/misc/bi_traits.hpp>
#include <data/core/utils/misc/utils.hpp>

#include <support/bi_rounding.hpp>

namespace BatmanInfer {
    namespace utils {
        namespace cast {
            // *INDENT-OFF*
// clang-format off
// same type
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_same<T, U>::value, int>::type = 0>
            T saturate_cast(U v) {
                return v;
            }

// signed -> signed widening/same_width
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_signed<U>() &&
                                            std::is_signed<T>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) >= sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(v);
            }

// signed -> signed narrowing
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_signed<U>() &&
                                            std::is_signed<T>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) < sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(misc::utility::clamp<U>(v, std::numeric_limits<T>::lowest(),
                                                              std::numeric_limits<T>::max()));
            }

// unsigned -> signed widening
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_unsigned<U>() &&
                                            std::is_signed<T>() &&
                                            !std::is_same<T, U>::value &&
                                            (sizeof(T) > sizeof(U)),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(v);
            }

// unsigned -> signed narrowing
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_unsigned<U>() &&
                                            std::is_signed<T>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) < sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(std::min<U>(v, std::numeric_limits<T>::max()));
            }

// unsigned -> signed same_width
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_unsigned<U>() &&
                                            std::is_signed<T>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) == sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(std::min<U>(v, std::numeric_limits<T>::max()));
            }

// signed -> unsigned widening/same width
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_signed<U>() &&
                                            std::is_unsigned<T>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) >= sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(std::max<U>(0, v));
            }

// signed -> unsigned narrowing
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_signed<U>() &&
                                            std::is_unsigned<T>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) < sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(misc::utility::clamp<U>(v, 0, std::numeric_limits<T>::max()));
            }

// unsigned -> unsigned widening/same width
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_unsigned<T>() &&
                                            std::is_unsigned<U>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) >= sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(v);
            }

// unsigned -> unsigned narrowing
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value &&
                                            std::is_unsigned<T>() &&
                                            std::is_unsigned<U>() &&
                                            !std::is_same<T, U>::value &&
                                            sizeof(T) < sizeof(U),
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(misc::utility::clamp<U>(v, std::numeric_limits<T>::lowest(),
                                                              std::numeric_limits<T>::max()));
            }

// float -> int
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            traits::is_floating_point<U>::value,
                            int>::type = 0>
            inline T saturate_cast(U v) {
                int32_t vi = utils::rounding::round_half_away_from_zero(v);
                return saturate_cast<T>(vi);
            }

// float -> int
            template<typename T,
                    typename U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            traits::is_floating_point<U>::value,
                            int>::type = 0>
            inline T saturate_static_cast(U v) {
                int32_t vi = static_cast<int32_t>(v);
                return saturate_cast<T>(vi);
            }

// int -> float
            template<typename T,
                    typename U,
                    typename std::enable_if<traits::is_floating_point<T>::value &&
                                            std::is_integral<U>::value,
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(v);
            }

// float -> float
            template<typename T,
                    typename U,
                    typename std::enable_if<traits::is_floating_point<T>::value &&
                                            traits::is_floating_point<U>::value,
                            int>::type = 0>
            inline T saturate_cast(U v) {
                return static_cast<T>(v);
            }
// clang-format on
// *INDENT-ON*
        }
    }
}