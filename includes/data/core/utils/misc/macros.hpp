//
// Created by Mason on 2024/12/31.
//

#ifndef BATMANINFER_MACROS_HPP
#define BATMANINFER_MACROS_HPP

#if defined(__cplusplus) && (__cplusplus >= 201402L)

#define BI_COMPUTE_DEPRECATED                           [[deprecated]]
#define BI_COMPUTE_DEPRECATED_REL(rel)                  [[deprecated("Deprecated in : " #rel)]]
#define BI_COMPUTE_DEPRECATED_REL_REPLACE(rel, replace) [[deprecated("Deprecated in : " #rel " - Use : " #replace)]]

#elif defined(__GNUC__) || defined(__clang__)

#define BI_COMPUTE_DEPRECATED          __attribute__((deprecated))
#define BI_COMPUTE_DEPRECATED_REL(rel) __attribute__((deprecated("Deprecated in : " #rel)))
#define BI_COMPUTE_DEPRECATED_REL_REPLACE(rel, replace) \
    __attribute__((deprecated("Deprecated in : " #rel " - Use : " #replace)))

#else // defined(__cplusplus) && (__cplusplus >= 201402L)

#define BI_COMPUTE_DEPRECATED
#define BI_COMPUTE_DEPRECATED_REL(rel)
#define BI_COMPUTE_DEPRECATED_REL_REPLACE(rel, replace)

#endif // defined(__cplusplus) && (__cplusplus >= 201402L)

#endif //BATMANINFER_MACROS_HPP
