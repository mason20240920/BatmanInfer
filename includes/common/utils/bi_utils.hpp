//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_COMMON_BI_UTILS_HPP
#define BATMANINFER_COMMON_BI_UTILS_HPP

#include <algorithm>
#include <initializer_list>
#include <type_traits>

namespace BatmanInfer {
    namespace utils {
        /**
         * 转换一个强类型枚举到老的C枚举
         * @tparam E Plain old C enum
         * @tparam SE Strongly typed resulting enum
         * @param v value to convert
         * @return A corresponding plain old C enumeration
         */
        template<typename E, typename SE>
        constexpr E as_cenum(const SE v) noexcept {
            return static_cast<E>(static_cast<std::underlying_type_t<SE>>(v));
        }

        /**
         * 与上面相反
         * @tparam SE
         * @tparam E
         * @param val
         * @return
         */
        template<typename SE, typename E>
        constexpr SE as_enum(const E val) noexcept {
            return static_cast<SE>(val);
        }

        template<typename E>
        bool is_in(E check, std::initializer_list<E> list) {
            return std::any_of(list.begin(),
                               list.end(),
                               [&check](E e) { return check == e; });
        }
    }
}

#endif //BATMANINFER_COMMON_BI_UTILS_HPP
