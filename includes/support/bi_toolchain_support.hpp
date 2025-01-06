//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_TOOLCHAIN_SUPPORT_HPP
#define BATMANINFER_BI_TOOLCHAIN_SUPPORT_HPP

#include "support/b_float16.hpp"
#include "support/bi_half.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // M_PI

namespace BatmanInfer {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    typedef __fp16 float16_t;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    namespace support {
        namespace cpp11 {
#if (__ANDROID__ || BARE_METAL)
            template <typename T>
            inline T nearbyint(T value)
            {
                return static_cast<T>(::nearbyint(value));
            }

            /**
             * Round floating-point value with half value rounding away from zero.
             * @note This function implements the same behaviour as std::round except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains
             *
             * @param[in] value floating-point value to be rounded.
             * @return Floating-point value of rounded @p value.
             *
             */
            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T round(T value)
            {
                return ::round(value);
            }

            /**
             * Round floating-point value with half value rounding away from zero and cast to long
             *
             * @note This function implements the same behaviour as std::lround except that it doesn't
             *       support Integral type. The latter is not in the namespace std in some Android toolchains.
             *
             * @param[in] value floating-point value to be rounded.
             *
             * @return Floating-point value of rounded @p value casted to long
             */
            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline long lround(T value)
            {
                return ::lround(value);
            }

            /**
             * Truncate floating-point value.
              * std::trunc(x) =
              *   { floor(x), if x >= 0
              *   { ceil(x),  if x < 0
              *
              * Example:
              *   std::trunc( 3.9) =  3
              *   std::trunc(-3.9) = -3
              *   std::trunc( 2.0) =  2
              *   std::trunc(-2.0) = -2
              *   std::trunc( 0.0) =  0
              * @note This function implements the same behaviour as std::truncate except that it doesn't
              *       support Integral type. The latter is not in the namespace std in some Android toolchains.
              *
              * @param[in] value floating-point value to be truncated.
              *
              * @return Floating-point value of truncated @p value.
             */
            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T trunc(T value)
            {
                return ::trunc(value);
            }


            /**
             *  @brief Composes a floating point value with the magnitude of @p x and the sign of @p y.
             *
             *  @note This function implements the same behaviour as std::copysign except that it doesn't
             *        support Integral type. The latter is not in the namespace std in some Android toolchains.
             *
             *  @param[in] x value that contains the magnitude to be used in constructing the result.
             *  @param[in] y value that contains the sign to be used in construct in the result.
             *
             *  @return Floating-point value with magnitude of @p x and sign of @p y.
             */
            template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T copysign(T x, T y)
            {
                return ::copysign(x, y);
            }

            template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value
#ifdef ___ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    || std::is_same<T, float16_t>::value
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        >::type>
            inline T fma(T x, T y, T z)
            {
                return ::fma(x, y, z);
            }

            /**
             * Loads the data from the given location, converts them to character string equivalents
             * and writes the result to a character string buffer.
             *
             * @param[in] s    Pointer to a character string to write to
             * @param[in] n    Up to buf_size - 1 characters may be written, plus the null ending character
             * @param[in] fmt  Pointer to a null-ended multibyte string specifying how to interpret the data.
             * @param[in] args Arguments forwarded to snprintf.
             *
             * @return  Number of characters that would have been written for a sufficiently large buffer
             *          if successful (not including the ending null character), or a negative value if an error occurred.
             */
            template<typename... Ts>
            inline int snprintf(char *s,
                               size_t n,
                               const char* fmt,
                               Ts &&...args) {
                return ::snprint(s, n, fmt, std::forward<Ts>(args)...);
            }
#else

            /**
             * @brief Round floating-point value with half value rounding away from zero.
             *
             * @note This function implements the same behaviour as std::round except that it doesn't
             *       support Integral type. The latter is not in the namespace std in some Android toolchains.
             * @tparam T
             * @param value floating-point value to be rounded.
             * @return  Floating-point value of rounded @p value.
             */
            template<typename T>
            inline T nearbyint(T value) {
                return static_cast<T>(std::nearbyint(value));
            }

            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T round(T value) {
                return std::round(value);
            }

            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T lround(T value) {
                return std::lround(value);
            }

            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T trunc(T value) {
                return std::trunc(value);
            }

            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
            inline T copysign(T x, T y) {
                return std::copysign(x, y);
            }

            template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value
                                                                    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                                    || std::is_same<T, float16_t>::value

#endif
            >::type>
            inline T fma(T x, T y, T z) {
                return std::fma(x, y, z);
            }

            template<typename... Ts>
            inline int snprintf(char *s,
                                std::size_t n,
                                const char *fmt,
                                Ts &&...args) {
                return std::snprintf(s, n, fmt, std::forward<Ts>(args)...);
            }

#endif

            template<typename T>
            inline T lowest() {
                return std::numeric_limits<T>::lowest();
            }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

            template<>
            inline __fp16 lowest<__fp16>() {
                return std::numeric_limits<half_float::half>::lowest();
            }

#endif

            template<>
            inline bfloat16 lowest<bfloat16>() {
                return bfloat16::lowest();
            }

            template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
            inline bool isfinite(T value) {
                return std::isfinite(static_cast<double>(value));
            }

            inline bool isfinite(half_float::half value) {
                return half_float::isfinite(value);
            }

            inline bool isfinite(bfloat16 value) {
                return std::isfinite(float(value));
            }

            template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
            inline bool signbit(T value) {
                return std::signbit(static_cast<double>(value));
            }

            inline bool signbit(half_float::half value) {
                return half_float::signbit(value);
            }

            inline bool signbit(bfloat16 value) {
                return std::signbit(float(value));
            }

        }

    }
}

#endif //BATMANINFER_BI_TOOLCHAIN_SUPPORT_HPP
