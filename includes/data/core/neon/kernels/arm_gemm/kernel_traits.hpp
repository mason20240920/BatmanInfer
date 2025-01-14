//
// Created by Mason on 2025/1/14.
//

#pragma once

namespace BatmanGemm {

    namespace {
        template<class T>
        constexpr auto is_sme_impl(int)
        -> decltype(T::is_sme(), std::true_type{}) {
            return std::true_type{};
        }

        template<class>
        constexpr auto is_sme_impl(...) -> std::false_type {
            return std::false_type{};
        }
    }

    template<class T>
    struct is_sme {
        static constexpr auto value = std::is_same<decltype(is_sme_impl<T>(0)),
                std::true_type>::value;
    };

}  // namespace BatmanGemm
