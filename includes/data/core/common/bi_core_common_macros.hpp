//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_CORE_COMMON_MACROS_HPP
#define BATMANINFER_BI_CORE_COMMON_MACROS_HPP

#define BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(TypeName) \
    TypeName(const TypeName &)            = delete;    \
    TypeName &operator=(const TypeName &) = delete;    \
    TypeName(TypeName &&)                 = default;   \
    TypeName &operator=(TypeName &&)      = default

#endif //BATMANINFER_BI_CORE_COMMON_MACROS_HPP
