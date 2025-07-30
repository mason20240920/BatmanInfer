//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_UTILS_HPP
#define BATMANINFER_BI_UTILS_HPP

#include <data/core/bi_types.hpp>
#include <support/bi_toolchain_support.hpp>
#include "data/core/bi_error.h"
#include "data/core/utils/data_type_utils.hpp"

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
    template<typename T>
    int max_consecutive_elements_display_width_impl(std::ostream &s, const T *ptr, unsigned int n) {
        using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;

        int max_width = -1;
        for (unsigned int i = 0; i < n; ++i) {
            std::stringstream ss;
            ss.copyfmt(s);

            if (std::is_same<typename std::decay<T>::type, half>::value) {
                // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
                ss << static_cast<T>(ptr[i]);
            } else if (std::is_same<typename std::decay<T>::type, bfloat16>::value) {
                // We use T instead of print_type here is because the std::is_floating_point<bfloat> returns false and then the print_type becomes int.
                ss << float(ptr[i]);
            } else {
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
    template<typename T>
    void print_consecutive_elements_impl(
            std::ostream &s, const T *ptr, unsigned int n, int stream_width = 0,
            const std::string &element_delim = " ") {
        using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;
        std::ios stream_status(nullptr);
        stream_status.copyfmt(s);

        for (unsigned int i = 0; i < n; ++i) {
            // Set stream width as it is not a "sticky" stream manipulator
            if (stream_width != 0) {
                s.width(stream_width);
            }

            if (std::is_same<typename std::decay<T>::type, half>::value) {
                // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
                s << std::right << static_cast<T>(ptr[i]) << element_delim;
            } else if (std::is_same<typename std::decay<T>::type, bfloat16>::value) {
                // We use T instead of print_type here is because the std::is_floating_point<bfloat16> returns false and then the print_type becomes int.
                s << std::right << float(ptr[i]) << element_delim;
            } else {
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
    void print_consecutive_elements(std::ostream &s,
                                    BIDataType dt,
                                    const uint8_t *ptr,
                                    unsigned int n,
                                    int stream_width,
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

#endif // BI_COMPUTE_ASSERTS_ENABLED

/** Returns expected width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 * @param[in] dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
    std::pair<unsigned int, unsigned int> scaled_dimensions(int width,
                                                            int height,
                                                            int kernel_width,
                                                            int kernel_height,
                                                            const BIPadStrideInfo &pad_stride_info,
                                                            const Size2D &dilation = Size2D(1U, 1U));

    /** Returns calculated width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second, returned values can be < 1
 */
    std::pair<int, int> scaled_dimensions_signed(
            int width, int height, int kernel_width, int kernel_height, const BIPadStrideInfo &pad_stride_info);

/** Returns expected width and height of the deconvolution's output tensor.
 *
 * @param[in] in_width        Width of input tensor (Number of columns)
 * @param[in] in_height       Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
    std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int in_width,
                                                                          unsigned int in_height,
                                                                          unsigned int kernel_width,
                                                                          unsigned int kernel_height,
                                                                          const BIPadStrideInfo &pad_stride_info);

/** Returns output quantization information for softmax layer
 *
 * @param[in] input_type The data type of the input tensor
 * @param[in] is_log     True for log softmax
 *
 * @return Quantization information for the output tensor
 */
    BIQuantizationInfo get_softmax_output_quantization_info(BIDataType input_type, bool is_log);

    template<typename T>
    inline void permute_strides(BIDimensions<T> &dimensions, const PermutationVector &perm) {
        const auto old_dim = misc::utility::make_array<BIDimensions<T>::num_max_dimensions>(dimensions.begin(),
                                                                                            dimensions.end());
        for (unsigned int i = 0; i < perm.num_dimensions(); ++i) {
            T dimension_val = old_dim[i];
            dimensions.set(perm[i], dimension_val);
        }
    }

    /**
     * Returns a pair of minimum and maximum values for a quantized activation
     * @param act_info The information for activation
     * @param data_type The used data type
     * @param oq_info The output quantization information
     * @return
     */
    std::pair<int32_t, int32_t> get_quantized_activation_min_max(const BIActivationLayerInfo &act_info,
                                                                 BIDataType data_type,
                                                                 BIUniformQuantizationInfo oq_info);

    /**
     * @brief 计算一个vector中从开头到指定索引（包含该索引）的所有元素的和
     *
     * @param vec
     * @param index
     * @return size_t  返回从索引0到index所有元素的和。
     */
    inline size_t get_vec_sum(const std::vector<size_t> &vec,const unsigned index, const unsigned int max_seq) {
        BI_COMPUTE_ERROR_ON_MSG(index >= vec.size(), "Index is out of range of the vector.");

        return max_seq * index + vec[index];
    }

    inline size_t get_remain_seq_sum_minus_one(const std::vector<size_t> &vec, const unsigned index) {
        // 1. 边界检查
        BI_COMPUTE_ERROR_ON_MSG(index > vec.size(), "Index is out of range of the vector.");

        // 2. 计算从 vec.begin() 到 vec.begin() + index 的总和
        size_t original_sum = std::accumulate(vec.begin(), vec.begin() + index, size_t{0});

        // 3. 减去元素的数量 (index + 1)
        //    需要检查确保和不会下溢（size_t 是无符号的）
        const size_t count = index;
        if (original_sum < count) {
            // 根据业务逻辑处理下溢情况，可以抛出异常或返回0
            throw std::logic_error("Sum is less than the number of elements, would underflow.");
        }

        return original_sum - count;
    }

    inline size_t get_remain_seq_sum(const std::vector<size_t> &vec, const unsigned index) {
        // 1. 边界检查
        BI_COMPUTE_ERROR_ON_MSG(index > vec.size(), "Index is out of range of the vector.");

        // 2. 计算从 vec.begin() 到 vec.begin() + index 的总和
        size_t original_sum = std::accumulate(vec.begin(), vec.begin() + index, size_t{0});

        return original_sum;
    }
}

#endif //BATMANINFER_BI_UTILS_HPP
