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
#include "function_info/bi_MatMulInfo.h"
#include "runtime/neon/functions/bi_ne_mat_mul.hpp"
#include "function_info/bi_fullyConnectedLayerInfo.h"

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

    inline std::string to_string(const BIRoundingPolicy &rounding_policy) {
        std::stringstream str;
        str << rounding_policy;
        return str.str();
    }

    /** Formatted output of the GEMMLowpOutputStageType type.
     *
     * @param[out] os        Output stream.
     * @param[in]  gemm_type GEMMLowpOutputStageType to output.
     *
     * @return Modified output stream.
     */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIGEMMLowpOutputStageType &gemm_type) {
        switch (gemm_type) {
            case BIGEMMLowpOutputStageType::NONE:
                os << "NONE";
                break;
            case BIGEMMLowpOutputStageType::QUANTIZE_DOWN:
                os << "QUANTIZE_DOWN";
                break;
            case BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
                os << "QUANTIZE_DOWN_FIXEDPOINT";
                break;
            case BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
                os << "QUANTIZE_DOWN_FLOAT";
                break;
            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }
        return os;
    }

    /** Formatted output of the DataType type.
*
* @param[out] os        Output stream.
* @param[in]  data_type Type to output.
*
* @return Modified output stream.
*/
    inline ::std::ostream &operator<<(::std::ostream &os, const BIDataType &data_type) {
        switch (data_type) {
            case BIDataType::UNKNOWN:
                os << "UNKNOWN";
                break;
            case BIDataType::U8:
                os << "U8";
                break;
            case BIDataType::QSYMM8:
                os << "QSYMM8";
                break;
            case BIDataType::QASYMM8:
                os << "QASYMM8";
                break;
            case BIDataType::QASYMM8_SIGNED:
                os << "QASYMM8_SIGNED";
                break;
            case BIDataType::QSYMM8_PER_CHANNEL:
                os << "QSYMM8_PER_CHANNEL";
                break;
            case BIDataType::S8:
                os << "S8";
                break;
            case BIDataType::U16:
                os << "U16";
                break;
            case BIDataType::S16:
                os << "S16";
                break;
            case BIDataType::QSYMM16:
                os << "QSYMM16";
                break;
            case BIDataType::QASYMM16:
                os << "QASYMM16";
                break;
            case BIDataType::U32:
                os << "U32";
                break;
            case BIDataType::S32:
                os << "S32";
                break;
            case BIDataType::U64:
                os << "U64";
                break;
            case BIDataType::S64:
                os << "S64";
                break;
            case BIDataType::BFLOAT16:
                os << "BFLOAT16";
                break;
            case BIDataType::F16:
                os << "F16";
                break;
            case BIDataType::F32:
                os << "F32";
                break;
            case BIDataType::F64:
                os << "F64";
                break;
            case BIDataType::SIZET:
                os << "SIZET";
                break;
            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

    /** Formatted output of the DataType type.
    *
    * @param[in] data_type Type to output.
    *
    * @return Formatted string.
    */
    inline std::string to_string(const BatmanInfer::BIDataType &data_type) {
        std::stringstream str;
        str << data_type;
        return str.str();
    }

    /** Formatted output of the BIGEMMLowpOutputStageInfo type.
     *
     * @param[out] os        Output stream.
     * @param[in]  gemm_info BIGEMMLowpOutputStageInfo to output.
     *
     * @return Modified output stream.
     */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIGEMMLowpOutputStageInfo &gemm_info) {
        os << "{type=" << gemm_info.type << ", "
           << "gemlowp_offset=" << gemm_info.gemmlowp_offset << ", "
           << "gemmlowp_multiplier=" << gemm_info.gemmlowp_multiplier << ", "
           << "gemmlowp_shift=" << gemm_info.gemmlowp_shift << ", "
           << "gemmlowp_min_bound=" << gemm_info.gemmlowp_min_bound << ", "
           << "gemmlowp_max_bound=" << gemm_info.gemmlowp_max_bound << ", "
           << "gemmlowp_multipliers=" << gemm_info.gemmlowp_multiplier << ", "
           << "gemmlowp_shifts=" << gemm_info.gemmlowp_shift << ", "
           << "gemmlowp_real_multiplier=" << gemm_info.gemmlowp_real_multiplier << ", "
           << "is_quantized_per_channel=" << gemm_info.is_quantized_per_channel << ", "
           << "output_data_type=" << gemm_info.output_data_type << "}";
        return os;
    }

    /** Converts a @ref BIGEMMLowpOutputStageInfo to string
     *
     * @param[in] gemm_info GEMMLowpOutputStageInfo value to be converted
     *
     * @return String representing the corresponding GEMMLowpOutputStageInfo
     */
    inline std::string to_string(const BIGEMMLowpOutputStageInfo &gemm_info) {
        std::stringstream str;
        str << gemm_info;
        return str.str();
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
    inline ::std::ostream &operator<<(::std::ostream &os, const BIQuantizationInfo &qinfo) {
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
    inline std::string to_string(const BIQuantizationInfo &quantization_info) {
        std::stringstream str;
        str << quantization_info;
        return str.str();
    }

    inline ::std::ostream &operator<<(::std::ostream &os, const BIConvertPolicy &policy) {
        switch (policy) {
            case BIConvertPolicy::WRAP:
                os << "WRAP";
                break;
            case BIConvertPolicy::SATURATE:
                os << "SATURATE";
                break;
            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }


    inline std::string to_string(const BIConvertPolicy &policy) {
        std::stringstream str;
        str << policy;
        return str.str();
    }

    inline ::std::ostream &
    operator<<(::std::ostream &os, const BIActivationLayerInfo::ActivationFunction &act_function) {
        switch (act_function) {
            case BIActivationLayerInfo::ActivationFunction::ABS:
                os << "ABS";
                break;
            case BIActivationLayerInfo::ActivationFunction::LINEAR:
                os << "LINEAR";
                break;
            case BIActivationLayerInfo::ActivationFunction::LOGISTIC:
                os << "LOGISTIC";
                break;
            case BIActivationLayerInfo::ActivationFunction::RELU:
                os << "RELU";
                break;
            case BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                os << "BOUNDED_RELU";
                break;
            case BIActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                os << "LEAKY_RELU";
                break;
            case BIActivationLayerInfo::ActivationFunction::SOFT_RELU:
                os << "SOFT_RELU";
                break;
            case BIActivationLayerInfo::ActivationFunction::SQRT:
                os << "SQRT";
                break;
            case BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                os << "LU_BOUNDED_RELU";
                break;
            case BIActivationLayerInfo::ActivationFunction::ELU:
                os << "ELU";
                break;
            case BIActivationLayerInfo::ActivationFunction::SQUARE:
                os << "SQUARE";
                break;
            case BIActivationLayerInfo::ActivationFunction::TANH:
                os << "TANH";
                break;
            case BIActivationLayerInfo::ActivationFunction::IDENTITY:
                os << "IDENTITY";
                break;
            case BIActivationLayerInfo::ActivationFunction::HARD_SWISH:
                os << "HARD_SWISH";
                break;
            case BIActivationLayerInfo::ActivationFunction::SWISH:
                os << "SWISH";
                break;
            case BIActivationLayerInfo::ActivationFunction::GELU:
                os << "GELU";
                break;

            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

/** Formatted output of the activation function info type.
 *
 * @param[in] info ActivationLayerInfo to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BatmanInfer::BIActivationLayerInfo &info) {
        std::stringstream str;
        if (info.enabled()) {
            str << info.activation();
        }
        return str.str();
    }

    inline ::std::ostream &operator<<(::std::ostream &os, const GEMMInfo &info) {
        os << "{is_a_reshaped=" << info.is_a_reshaped() << ",";
        os << "is_b_reshaped=" << info.is_b_reshaped() << ",";
        os << "reshape_b_only_on_first_run=" << info.reshape_b_only_on_first_run() << ",";
        os << "depth_output_gemm3d=" << info.depth_output_gemm3d() << ",";
        os << "reinterpret_input_as_3d=" << info.reinterpret_input_as_3d() << ",";
        os << "retain_internal_weights=" << info.retain_internal_weights() << ",";
        os << "fp_mixed_precision=" << info.fp_mixed_precision() << ",";
        os << "broadcast_bias=" << info.broadcast_bias() << ",";
        os << "pretranspose_B=" << info.pretranspose_B() << ",";
        os << "}";

        return os;
    }

    inline std::string to_string(const GEMMInfo &info) {
        std::stringstream str;
        str << info;
        return str.str();
    }

    /** Formatted output of the Coordinates type.
     *
     * @param[in] coord Type to output.
     *
     * @return Formatted string.
    */
    inline std::string to_string(const BICoordinates &coord) {
        std::stringstream str;
        str << coord;
        return str.str();
    }

    /** Formatted output of the arm_compute::MatMulInfo type.
     *
     * @param[out] os          Output stream.
     * @param[in]  matmul_info arm_compute::MatMulInfo  type to output.
     *
     * @return Modified output stream.
     */
    inline ::std::ostream &operator<<(::std::ostream &os, const BatmanInfer::BIMatMulInfo &matmul_info) {
        os << "MatMulKernelInfo="
           << "["
           << "adj_lhs=" << matmul_info.adj_lhs() << ", "
           << "adj_rhs=" << matmul_info.adj_rhs() << "] ";
        return os;
    }

    /** Formatted output of the arm_compute::MatMulInfo type.
     *
     * @param[in] matmul_info arm_compute::MatMulInfo type to output.
     *
     * @return Formatted string.
    */
    inline std::string to_string(const BatmanInfer::BIMatMulInfo &matmul_info) {
        std::stringstream str;
        str << matmul_info;
        return str.str();
    }

    /** Formatted output of the arm_compute::CpuMatMulSettings type.
     *
     * @param[out] os       Output stream.
     * @param[in]  settings arm_compute::CpuMatMulSettings type to output.
     *
     * @return Modified output stream.
     */
    inline ::std::ostream &operator<<(::std::ostream &os, const BatmanInfer::BICpuMatMulSettings &settings) {
        os << "CpuMatMulSettings="
           << "["
           << "fast_math=" << settings.fast_math() << ",fixed_format=" << settings.fixed_format() << "]";

        return os;
    }

    /** Formatted output of the arm_compute::CpuMatMulSettings type.
     *
     * @param[in] settings arm_compute::CpuMatMulSettings type to output.
     *
     * @return Formatted string.
     */
    inline std::string to_string(const BatmanInfer::BICpuMatMulSettings &settings) {
        std::stringstream str;
        str << settings;
        return str.str();
    }

    /** Formatted output of the Strides type.
     *
     * @param[in] stride Type to output.
     *
     * @return Formatted string.
     */
    inline std::string to_string(const BIStrides &stride) {
        std::stringstream str;
        str << stride;
        return str.str();
    }

    /** [Print DataLayout type] **/
    /** Formatted output of the DataLayout type.
     *
     * @param[out] os          Output stream.
     * @param[in]  data_layout Type to output.
     *
     * @return Modified output stream.
     */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIDataLayout &data_layout) {
        switch (data_layout) {
            case BIDataLayout::UNKNOWN:
                os << "UNKNOWN";
                break;
            case BIDataLayout::NHWC:
                os << "NHWC";
                break;
            case BIDataLayout::NCHW:
                os << "NCHW";
                break;
            case BIDataLayout::NDHWC:
                os << "NDHWC";
                break;
            case BIDataLayout::NCDHW:
                os << "NCDHW";
                break;
            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

/** Formatted output of the DataLayout type.
 *
 * @param[in] data_layout Type to output.
 *
 * @return Formatted string.
 */
    inline std::string to_string(const BatmanInfer::BIDataLayout &data_layout) {
        std::stringstream str;
        str << data_layout;
        return str.str();
    }

    /** Formatted output of the TensorShape type.
     *
     * @param shape
     * @return
     */
    inline std::string to_string(const BITensorShape &shape) {
        std::stringstream str;
        str << shape;
        return str.str();
    }

    inline ::std::ostream &operator<<(::std::ostream &os, const BIFullyConnectedLayerInfo &layer_info) {
        os << "{activation_info=" << to_string(layer_info.activation_info) << ", "
           << "weights_trained_layout=" << layer_info.weights_trained_layout << ", "
           << "transpose_weights=" << layer_info.transpose_weights << ", "
           << "are_weights_reshaped=" << layer_info.are_weights_reshaped << ", "
           << "retain_internal_weights=" << layer_info.retain_internal_weights << ", "
           << "fp_mixed_precision=" << layer_info.fp_mixed_precision << "}";
        return os;
    }

    /** Converts a @ref FullyConnectedLayerInfo to string
     *
     * @param[in] info FullyConnectedLayerInfo value to be converted
     *
     * @return String  representing the corresponding FullyConnectedLayerInfo
     */
    inline std::string to_string(const BIFullyConnectedLayerInfo &info) {
        std::stringstream str;
        str << info;
        return str.str();
    }

}

#endif //BATMANINFER_BI_TYPE_PRINTER_HPP
