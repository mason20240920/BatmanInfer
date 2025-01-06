//
// Created by Mason on 2025/1/6.
//

#ifndef BATMANINFER_BI_VALIDATE_HPP
#define BATMANINFER_BI_VALIDATE_HPP

#include <data/core/bi_error.h>
#include <data/core/bi_i_kernel.hpp>

#include <algorithm>

namespace BatmanInfer {
    /**
     * @brief 如果其中一个指针是空指针，则创建一个错误。
     * @tparam Ts
     * @param function 发生错误的函数。
     * @param file Name of the file where the error occurred.
     * @param line Line on which the error occurred.
     * @param pointers Pointers to check against nullptr.
     * @return
     */
    template<typename ... Ts>
    inline BatmanInfer::BIStatus error_on_nullptr(const char *function,
                                                  const char *file,
                                                  const int line,
                                                  Ts &&...pointers) {
        const std::array<const void *, sizeof...(Ts)> pointers_array{{std::forward<Ts>(pointers)...}};
        bool                                          has_nullptr = std::any_of(pointers_array.begin(),
                                                                                pointers_array.end(),
                                                                                [&](const void *ptr) {
                                                                                    return (ptr == nullptr);
                                                                                });

        BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(has_nullptr, function, file, line, "Nullptr object!");
        return BatmanInfer::BIStatus{};
    }

#define BI_COMPUTE_ERROR_ON_NULLPTR(...) \
    BI_COMPUTE_ERROR_THROW_ON(::BatmanInfer::error_on_nullptr(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(...) \
    BI_COMPUTE_RETURN_ON_ERROR(::BatmanInfer::error_on_nullptr(__func__, __FILE__, __LINE__, __VA_ARGS__))
}

#endif //BATMANINFER_BI_VALIDATE_HPP
