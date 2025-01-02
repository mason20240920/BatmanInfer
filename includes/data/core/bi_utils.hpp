//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_UTILS_HPP
#define BATMANINFER_BI_UTILS_HPP

#include <data/core/bi_types.hpp>
#include <support/bi_toolchain_support.hpp>

#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace BatmanInfer {
#ifdef BI_COMPUTE_ASSERTS_ENABLED
    /** Identify the maximum width of n consecutive elements.
 *
 * @param[in] s   The output stream which will be used to print the elements. Used to extract the stream format.
 * @param[in] ptr Pointer to the elements.
 * @param[in] n   Number of elements.
 *
 * @return The maximum width of the elements.
 */
    template <typename T>
    int max_consecutive_elements_display_width_impl(std::ostream &s, const T *ptr, unsigned int n)
    {
        using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;

        int max_width = -1;
        for (unsigned int i = 0; i < n; ++i)
        {
            std::stringstream ss;
            ss.copyfmt(s);

            if (std::is_same<typename std::decay<T>::type, half>::value)
            {
                // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
                ss << static_cast<T>(ptr[i]);
            }
            else if (std::is_same<typename std::decay<T>::type, bfloat16>::value)
            {
                // We use T instead of print_type here is because the std::is_floating_point<bfloat> returns false and then the print_type becomes int.
                ss << float(ptr[i]);
            }
            else
            {
                ss << static_cast<print_type>(ptr[i]);
            }

            max_width = std::max<int>(max_width, ss.str().size());
        }
        return max_width;
    }

    /** Print consecutive elements to an output stream.
 *
 * @param[out] s             Output stream to print the elements to.
 * @param[in]  ptr           Pointer to print the elements from.
 * @param[in]  n             Number of elements to print.
 * @param[in]  stream_width  (Optional) Width of the stream. If set to 0 the element's width is used. Defaults to 0.
 * @param[in]  element_delim (Optional) Delimeter among the consecutive elements. Defaults to space delimeter
 */
    template <typename T>
    void print_consecutive_elements_impl(
            std::ostream &s, const T *ptr, unsigned int n, int stream_width = 0, const std::string &element_delim = " ")
    {
        using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;
        std::ios stream_status(nullptr);
        stream_status.copyfmt(s);

        for (unsigned int i = 0; i < n; ++i)
        {
            // Set stream width as it is not a "sticky" stream manipulator
            if (stream_width != 0)
            {
                s.width(stream_width);
            }

            if (std::is_same<typename std::decay<T>::type, half>::value)
            {
                // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
                s << std::right << static_cast<T>(ptr[i]) << element_delim;
            }
            else if (std::is_same<typename std::decay<T>::type, bfloat16>::value)
            {
                // We use T instead of print_type here is because the std::is_floating_point<bfloat16> returns false and then the print_type becomes int.
                s << std::right << float(ptr[i]) << element_delim;
            }
            else
            {
                s << std::right << static_cast<print_type>(ptr[i]) << element_delim;
            }
        }

        // Restore output stream flags
        s.copyfmt(stream_status);
    }

    /** Print consecutive elements to an output stream.
 *
 * @param[out] s             Output stream to print the elements to.
 * @param[in]  dt            Data type of the elements
 * @param[in]  ptr           Pointer to print the elements from.
 * @param[in]  n             Number of elements to print.
 * @param[in]  stream_width  (Optional) Width of the stream. If set to 0 the element's width is used. Defaults to 0.
 * @param[in]  element_delim (Optional) Delimeter among the consecutive elements. Defaults to space delimeter
 */
void print_consecutive_elements(std::ostream      &s,
                                BIDataType           dt,
                                const uint8_t     *ptr,
                                unsigned int       n,
                                int                stream_width,
                                const std::string &element_delim = " ");

    /** Identify the maximum width of n consecutive elements.
 *
 * @param[in] s   Output stream to print the elements to.
 * @param[in] dt  Data type of the elements
 * @param[in] ptr Pointer to print the elements from.
 * @param[in] n   Number of elements to print.
 *
 * @return The maximum width of the elements.
 */
    int max_consecutive_elements_display_width(std::ostream &s, BIDataType dt, const uint8_t *ptr, unsigned int n);

#endif
}

#endif //BATMANINFER_BI_UTILS_HPP
