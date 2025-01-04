//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_UTILS_HPP
#define BATMANINFER_UTILS_HPP

#include <limits>

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

            /**
             * @brief 把字符串转为小写
             * @param string
             * @return
             */
            inline std::string tolower(std::string string) {
                std::transform(string.begin(), string.end(), string.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                return string;
            }

            inline std::string getenv(const std::string &env_name) {
#ifdef BARE_METAL
                BI_COMPUTE_UNUSED(env_name);
    return std::string{};
#else  // BARE_METAL
                const auto env_chr = std::getenv(env_name.c_str());
                return env_chr == nullptr ? std::string{} : std::string{env_chr};
#endif // BARE_METAL
            }
        }
    }
}

#endif //BATMANINFER_UTILS_HPP
