//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_STRING_SUPPORT_HPP
#define BATMANINFER_STRING_SUPPORT_HPP

#include <cassert>
#include <memory>
#include <sstream>
#include <string>

namespace BatmanInfer {
    namespace support {
        namespace cpp11 {
            enum class BINumericBase {
                BASE_10,
                BASE_16
            };

            /** Convert string values to integer.
             *
             * @note This function implements the same behaviour as std::stoi. The latter
             *       is missing in some Android toolchains.
             *
             * @param[in] str  String to be converted to int.
             * @param[in] pos  If idx is not a null pointer, the function sets the value of pos to the position of the first character in str after the number.
             * @param[in] base Numeric base used to interpret the string.
             *
             * @return Integer representation of @p str.
             */
            inline int stoi(const std::string &str, std::size_t *pos = 0, BINumericBase base = BINumericBase::BASE_10) {
                assert(base == BINumericBase::BASE_10 || base == BINumericBase::BASE_16);
                unsigned int x;
                std::stringstream ss;
                if (base == BINumericBase::BASE_16) {
                    ss << std::hex;
                }
                ss << str;
                ss >> x;

                if (pos) {
                    std::string s;
                    std::stringstream ss_p;

                    ss_p << x;
                    ss_p >> s;
                    *pos = s.length();
                }

                return x;
            }

            /** Convert string values to unsigned long.
             *
             * @note This function implements the same behaviour as std::stoul. The latter
             *       is missing in some Android toolchains.
             *
             * @param[in] str  String to be converted to unsigned long.
             * @param[in] pos  If idx is not a null pointer, the function sets the value of pos to the position of the first character in str after the number.
             * @param[in] base Numeric base used to interpret the string.
             *
             * @return Unsigned long representation of @p str.
             */
            inline unsigned long
            stoul(const std::string &str, std::size_t *pos = 0, BINumericBase base = BINumericBase::BASE_10) {
                assert(base == BINumericBase::BASE_10 || base == BINumericBase::BASE_16);
                std::stringstream stream;
                unsigned long value = 0;
                if (base == BINumericBase::BASE_16) {
                    stream << std::hex;
                }
                stream << str;
                stream >> value;

                if (pos) {
                    std::string s;
                    std::stringstream ss_p;

                    ss_p << value;
                    ss_p >> s;
                    *pos = s.length();
                }

                return value;
            }

#if (__ANDROID__ || BARE_METAL)
            /** Convert integer and float values to string.
            *
            * @note This function implements the same behaviour as std::to_string. The
            *       latter is missing in some Android toolchains.
            *
            * @param[in] value Value to be converted to string.
 *
 * @return String representation of @p value.
 */
template <typename T, typename std::enable_if<std::is_arithmetic<typename std::decay<T>::type>::value, int>::type = 0>
inline std::string to_string(T &&value)
{
    std::stringstream stream;
    stream << std::forward<T>(value);
    return stream.str();
}

// Specialization for const std::string&
inline std::string to_string(const std::string &value)
{
    return value;
}

/** Convert string values to float.
 *
 * @note This function implements the same behaviour as std::stof. The latter
 *       is missing in some Android toolchains.
 *
 * @param[in] str String to be converted to float.
 *
 * @return Float representation of @p str.
 */
inline float stof(const std::string &str)
{
    std::stringstream stream(str);
    float             value = 0.f;
    stream >> value;
    return value;
}

#else /* (__ANDROID__ || BARE_METAL) */

/** Convert integer and float values to string.
 *
 * @note This function acts as a convenience wrapper around std::to_string. The
 *       latter is missing in some Android toolchains.
 *
 * @param[in] value Value to be converted to string.
 *
 * @return String representation of @p value.
 */
            template<typename T>
            inline std::string to_string(T &&value) {
                return ::std::to_string(std::forward<T>(value));
            }

// Specialization for const std::string&
            inline std::string to_string(const std::string &value) {
                return value;
            }

/** Convert string values to float.
 *
 * @note This function acts as a convenience wrapper around std::stof. The
 *       latter is missing in some Android toolchains.
 *
 * @param[in] args Arguments forwarded to std::stof.
 *
 * @return Float representation of input string.
 */
            template<typename... Ts>
            int stof(Ts &&...args) {
                return ::std::stof(std::forward<Ts>(args)...);
            }

#endif /* (__ANDROID__ || BARE_METAL) */

            inline std::string to_string(bool value) {
                std::stringstream str;
                str << std::boolalpha << value;
                return str.str();
            }
        }
    }
}

#endif //BATMANINFER_STRING_SUPPORT_HPP
