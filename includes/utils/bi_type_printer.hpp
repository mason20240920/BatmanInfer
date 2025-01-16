//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_TYPE_PRINTER_HPP
#define BATMANINFER_BI_TYPE_PRINTER_HPP

#include <support/string_support.hpp>
#include <data/core/dimensions.hpp>
#include <data/core/bi_rounding.h>
#include <data/core/bi_types.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/utils/data_type_utils.hpp>

namespace BatmanInfer {
    /** Formatted output if arg is not null
 *
 * @param[in] arg Object to print
 *
 * @return String representing arg.
 */
    template<typename T>
    std::string to_string_if_not_null(T *arg) {
        if (arg == nullptr) {
            return "nullptr";
        } else {
            return to_string(*arg);
        }
    }

/** Fallback method: try to use std::to_string:
 *
 * @param[in] val Value to convert to string
 *
 * @return String representing val.
 */
    template<typename T>
    inline std::string to_string(const T &val) {
        return support::cpp11::to_string(val);
    }

/** Formatted output of a vector of objects.
 *
 * @note: Using the overloaded to_string() instead of overloaded operator<<(), because to_string() functions are
 *        overloaded for all types, where two or more of them can use the same operator<<(), ITensor is an example.
 *
 * @param[out] os   Output stream
 * @param[in]  args Vector of objects to print
 *
 * @return Modified output stream.
 */
    template<typename T>
    ::std::ostream &operator<<(::std::ostream &os, const std::vector<T> &args) {
        const size_t max_print_size = 5U;

        os << "[";
        bool first = true;
        size_t i;
        for (i = 0; i < args.size(); ++i) {
            if (i == max_print_size) {
                break;
            }
            if (first) {
                first = false;
            } else {
                os << ", ";
            }
            os << to_string(args[i]);
        }
        if (i < args.size()) {
            os << ", ...";
        }
        os << "]";
        return os;
    }

/** Formatted output of a vector of objects.
 *
 * @param[in] args Vector of objects to print
 *
 * @return String representing args.
 */
    template<typename T>
    std::string to_string(const std::vector<T> &args) {
        std::stringstream str;
        str << args;
        return str.str();
    }

/** Formatted output of the Dimensions type.
 *
 * @param[out] os         Output stream.
 * @param[in]  dimensions Type to output.
 *
 * @return Modified output stream.
 */
    template<typename T>
    inline ::std::ostream &operator<<(::std::ostream &os, const BIDimensions<T> &dimensions) {
        if (dimensions.num_dimensions() > 0) {
            os << dimensions[0];

            for (unsigned int d = 1; d < dimensions.num_dimensions(); ++d) {
                os << "," << dimensions[d];
            }
        }

        return os;
    }

/** Formatted output of the RoundingPolicy type.
 *
 * @param[out] os              Output stream.
 * @param[in]  rounding_policy Type to output.
 *
 * @return Modified output stream.
 */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIRoundingPolicy &rounding_policy) {
        switch (rounding_policy) {
            case BIRoundingPolicy::TO_ZERO:
                os << "TO_ZERO";
                break;
            case BIRoundingPolicy::TO_NEAREST_UP:
                os << "TO_NEAREST_UP";
                break;
            case BIRoundingPolicy::TO_NEAREST_EVEN:
                os << "TO_NEAREST_EVEN";
                break;
            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

/** Formatted output of the WeightsInfo type.
 *
 * @param[out] os           Output stream.
 * @param[in]  weights_info Type to output.
 *
 * @return Modified output stream.
 */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIWeightsInfo &weights_info) {
        os << weights_info.are_reshaped() << ";";
        os << weights_info.num_kernels() << ";" << weights_info.kernel_size().first << ","
           << weights_info.kernel_size().second;

        return os;
    }

/** Formatted output of the ROIPoolingInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  pool_info Type to output.
 *
 * @return Modified output stream.
 */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIROIPoolingLayerInfo &pool_info) {
        os << pool_info.pooled_width() << "x" << pool_info.pooled_height() << "~" << pool_info.spatial_scale();
        return os;
    }

/** Formatted output of the ROIPoolingInfo type.
 *
 * @param[in] pool_info Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BIROIPoolingLayerInfo &pool_info) {
        std::stringstream str;
        str << pool_info;
        return str.str();
    }

    /** Formatted output of the ITensorInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Tensor information.
 *
 * @return Modified output stream.
 */
    inline ::std::ostream &operator<<(std::ostream &os, const BIITensorInfo *info) {
        const BIDataType data_type = info->data_type();
//        const BIDataLayout data_layout = info->data_layout();

        os << "Shape=" << info->tensor_shape() << ","
           //           << "DataLayout=" << string_from_data_layout(data_layout) << ","
           << "DataType=" << string_from_data_type(data_type);

        if (is_data_type_quantized(data_type)) {
            const BIQuantizationInfo qinfo = info->quantization_info();
            const auto scales = qinfo.scale();
            const auto offsets = qinfo.offset();

            os << ", QuantizationInfo={"
               << "scales.size=" << scales.size() << ", scale(s)=" << scales << ", ";

            os << "offsets.size=" << offsets.size() << ", offset(s)=" << offsets << "}";
        }
        return os;
    }

/** Formatted output of the const TensorInfo& type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
    inline ::std::ostream &operator<<(::std::ostream &os, const BITensorInfo &info) {
        os << &info;
        return os;
    }

/** Formatted output of the const TensorInfo& type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BITensorInfo &info) {
        std::stringstream str;
        str << &info;
        return str.str();
    }

/** Formatted output of the const ITensorInfo& type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BIITensorInfo &info) {
        std::stringstream str;
        str << &info;
        return str.str();
    }

/** Formatted output of the const ITensorInfo* type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BIITensorInfo *info) {
        std::string ret_str = "nullptr";
        if (info != nullptr) {
            std::stringstream str;
            str << info;
            ret_str = str.str();
        }
        return ret_str;
    }

/** Formatted output of the ITensorInfo* type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(BIITensorInfo *info) {
        return to_string(static_cast<const BIITensorInfo *>(info));
    }

/** Formatted output of the ITensorInfo type obtained from const ITensor* type.
 *
 * @param[in] tensor Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BIITensor *tensor) {
        std::string ret_str = "nullptr";
        if (tensor != nullptr) {
            std::stringstream str;
            str << "ITensor->info(): " << tensor->info();
            ret_str = str.str();
        }
        return ret_str;
    }

    /** Formatted output of the ITensorInfo type obtained from the ITensor* type.
 *
 * @param[in] tensor Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(BIITensor *tensor) {
        return to_string(static_cast<const BIITensor *>(tensor));
    }

/** Formatted output of the ITensorInfo type obtained from the ITensor& type.
 *
 * @param[in] tensor Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(BIITensor &tensor) {
        std::stringstream str;
        str << "ITensor.info(): " << tensor.info();
        return str.str();
    }

    /** Formatted output of the QuantizationInfo type.
     *
     * @param[out] os    Output stream.
     * @param[in]  qinfo Type to output.
     *
     * @return Modified output stream.
     */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIQuantizationInfo &qinfo)
    {
        const BIUniformQuantizationInfo uqinfo = qinfo.uniform();
        os << "Scale:" << uqinfo.scale << "~";
        os << "Offset:" << uqinfo.offset;
        return os;
    }

    /** Formatted output of the QuantizationInfo type.
     *
     * @param[in] quantization_info Type to output.
     *
     * @return Formatted string.
     */
    inline std::string to_string(const BIQuantizationInfo &quantization_info)
    {
        std::stringstream str;
        str << quantization_info;
        return str.str();
    }

}

#endif //BATMANINFER_BI_TYPE_PRINTER_HPP
