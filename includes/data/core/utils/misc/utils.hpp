//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_UTILS_HPP
#define BATMANINFER_UTILS_HPP

#include "data/core/bi_error.h"

#include <limits>
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

namespace BatmanInfer {
    namespace misc {
        namespace utility {
            /**
             * @brief Base case of for_each.
             * 递归终止条件
             * @tparam F
             */
            template<typename F>
            inline void for_each(F &&) {

            }

            /**
             * @brief  Call the function for each of the arguments
             * @tparam F
             * @tparam T
             * @tparam Ts
             * @param func func Function to be called
             * @param argc arg  Argument passed to the function
             * @param args args Remaining arguments
             */
            template<typename F, typename T, typename... Ts>
            inline void for_each(F &&func,
                                 T &&argc,
                                 Ts &&... args) {
                func(std::forward<T>(argc));
                for_each(std::forward<F>(func), std::forward<Ts>(args)...);
            }

            /**
             * @brief 验证一个指针是否按照编译对齐了
             * @param ptr
             * @param alignment
             * @return
             */
            inline bool check_aligned(void *ptr,
                                      const size_t alignment) {
                return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
            }

            /**
             * @brief  generate_array继承自generate_array<T, N - 1, val, val, vals...>，即通过递归模板，
             *         将N逐步减小，直到N == 0时停止递归
             * @tparam T 数组中元素的类型
             * @tparam N 数组的大小
             * @tparam val  数组中每个元素的值
             * @tparam vals 可变参数模版，用于存储数组中的元素
             */
            template<typename T, std::size_t N, T val, T... vals>
            struct generate_array : generate_array<T, N - 1, val, val, vals...> {

            };

            /**
             * @brief 当N == 0时，递归终止，进入这个特化的模板
             * @tparam T 数组中元素的类型
             * @tparam val 数组中每个元素的值
             * @tparam vals 可变参数模版，用于存储数组中的元素
             */
            template<typename T, T val, T... vals>
            struct generate_array<T, 0, val, vals...> {
                static constexpr std::array<T, sizeof...(vals)> value{vals...};
            };

            template <std::size_t...>
            struct index_sequence
            {
            };

            template <std::size_t N, std::size_t... S>
            struct index_sequence_generator : index_sequence_generator<N - 1, N - 1, S...>
            {
            };

            template <std::size_t... S>
            struct index_sequence_generator<0u, S...> : index_sequence<S...>
            {
                using type = index_sequence<S...>;
            };

            template <std::size_t N>
            using index_sequence_t = typename index_sequence_generator<N>::type;

            template <typename T, T val, T... vals>
            constexpr std::array<T, sizeof...(vals)> generate_array<T, 0, val, vals...>::value;

            namespace detail
            {
                template <std::size_t... S,
                          typename Iterator,
                          typename T = std::array<typename std::iterator_traits<Iterator>::value_type, sizeof...(S)>>
                T make_array(Iterator first, index_sequence<S...>)
                {
                    return T{{first[S]...}};
                }
            } // namespace detail

            template <std::size_t N, typename Iterator>
            std::array<typename std::iterator_traits<Iterator>::value_type, N> make_array(Iterator first, Iterator last)
            {
                BI_COMPUTE_UNUSED(last);
                return detail::make_array(first, index_sequence_t<N>{});
            }

            /** Performs clamping among a lower and upper value.
             *
             * @param[in] n     Value to clamp.
             * @param[in] lower Lower threshold.
             * @param[in] upper Upper threshold.
             *
             *  @return Clamped value.
             */
            template <typename DataType, typename RangeType = DataType>
            inline DataType clamp(const DataType &n,
                                  const DataType &lower = std::numeric_limits<RangeType>::lowest(),
                                  const DataType &upper = std::numeric_limits<RangeType>::max())
            {
                return std::max(lower, std::min(n, upper));
            }

            /** Base case of foldl.
             *
             * @return value.
             */
            template <typename F, typename T>
            inline T &&foldl(F &&, T &&value)
            {
                return std::forward<T>(value);
            }

            /** Fold left.
             *
             * @param[in] func    Function to be called
             * @param[in] initial Initial value
             * @param[in] value   Argument passed to the function
             * @param[in] values  Remaining arguments
             */
            template <typename F, typename T, typename U, typename... Us>
            inline auto foldl(F &&func, T &&initial, U &&value, Us &&...values)
                -> decltype(func(std::forward<T>(initial), std::forward<U>(value)))
            {
                return foldl(std::forward<F>(func), func(std::forward<T>(initial), std::forward<U>(value)),
                             std::forward<Us>(values)...);
            }

            /** Perform an index sort of a given vector.
             *
             * @param[in] v Vector to sort
             *
             * @return Sorted index vector.
             */
            template <typename T>
            std::vector<size_t> sort_indices(const std::vector<T> &v)
            {
                std::vector<size_t> idx(v.size());
                std::iota(idx.begin(), idx.end(), 0);

                std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

                return idx;
            }

            /** 检查一个字符串是否包含给定后缀
             *
             * @param[in] str    Input string
             * @param[in] suffix Suffix to check for
             *
             * @return True if the string ends with the given suffix else false
             */
            inline bool endswith(const std::string &str, const std::string &suffix)
            {
                if (str.size() < suffix.size())
                {
                    return false;
                }
                return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
            }

            /** Convert string to lower case.
             *
             * @param[in] string To be converted string.
             *
             * @return Lower case string.
             */
            inline std::string tolower(std::string string)
            {
                std::transform(string.begin(), string.end(), string.begin(), [](unsigned char c) { return std::tolower(c); });
                return string;
            }

            /** Get environment variable as a string
             *
             * @note Return empty string on bare-metal
             *
             * @param[in] env_name Name of the Environment variable to retrieve
             *
             * @return Environment variable content, or empty string if the variable is undefined or on bare-metal
             */
            inline std::string getenv(const std::string &env_name)
            {
#ifdef BARE_METAL
                BI_COMPUTE_UNUSED(env_name);
                return std::string{};
#else  // BARE_METAL
                const auto env_chr = std::getenv(env_name.c_str());
                return env_chr == nullptr ? std::string{} : std::string{env_chr};
#endif // BARE_METAL
            }

        } // namespace utility
    } // namespace misc
} // namespace BatmanInfer

#endif //BATMANINFER_UTILS_HPP
