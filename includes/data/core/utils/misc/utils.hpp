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
        }
    }
}

#endif //BATMANINFER_UTILS_HPP
