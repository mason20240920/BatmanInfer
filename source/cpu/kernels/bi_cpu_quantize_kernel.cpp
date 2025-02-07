//
// Created by Mason on 2025/2/7.
//

#include <cpu/kernels/bi_cpu_quantize_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <common/bi_registers.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>
#include <cpu/kernels/quantize/generic/list.hpp>

#include <map>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {

                BIStatus validate_arguments(const BIITensorInfo *src, const BIITensorInfo *dst) {
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::F16, BIDataType::F32);
                    BI_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::QSYMM8, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::QASYMM16);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);

                    return BIStatus{};
                }

            } // namespace

            void BICpuQuantizeKernel::configure(const BIITensorInfo *src, BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

                static const std::map<std::string, QuantizeFunctionExecutorPtr> quant_map = {
                        {"op_QASYMM8_QASYMM8", REGISTER_INTEGER_NEON(u8_u8_run_quantize_qasymm8)},
                        {"op_QASYMM8_QASYMM8_SIGNED", REGISTER_INTEGER_NEON(u8_i8_run_quantize_qasymm8)},
                        {"op_QASYMM8_QASYMM16", REGISTER_INTEGER_NEON(u8_run_quantize_qasymm16)},

                        {"op_QASYMM8_SIGNED_QASYMM8", REGISTER_INTEGER_NEON(i8_u8_run_quantize_qasymm8)},
                        {"op_QASYMM8_SIGNED_QASYMM8_SIGNED", REGISTER_INTEGER_NEON(i8_i8_run_quantize_qasymm8)},
                        {"op_QASYMM8_SIGNED_QASYMM16", REGISTER_INTEGER_NEON(i8_run_quantize_qasymm16)},

                        // Functions for offset only requantization
                        {"op_OFFSET_ONLY_QASYMM8_QASYMM8", REGISTER_INTEGER_NEON(u8_u8_run_requantize_offset_only)},
                        {"op_OFFSET_ONLY_QASYMM8_QASYMM8_SIGNED",
                         REGISTER_INTEGER_NEON(u8_i8_run_requantize_offset_only)},
                        {"op_OFFSET_ONLY_QASYMM8_SIGNED_QASYMM8",
                         REGISTER_INTEGER_NEON(i8_u8_run_requantize_offset_only)},
                        {"op_OFFSET_ONLY_QASYMM8_SIGNED_QASYMM8_SIGNED",
                         REGISTER_INTEGER_NEON(i8_i8_run_requantize_offset_only)},

                        // Functions for offset uint8 to int8 and vice versa quantization (no scale changes)
                        {"op_OFFSET_ONLY_CONVERT_QASYMM8_SIGNED_QASYMM8",
                         REGISTER_INTEGER_NEON(i8_u8_run_requantize_offset_only_convert)},
                        {"op_OFFSET_ONLY_CONVERT_QASYMM8_QASYMM8_SIGNED",
                         REGISTER_INTEGER_NEON(u8_i8_run_requantize_offset_only_convert)},

                        {"op_F32_QSYMM8", REGISTER_FP32_NEON(fp32_i8_run_quantize_qsymm8)},
                        {"op_F32_QASYMM8", REGISTER_FP32_NEON(fp32_u8_run_quantize_qasymm8)},
                        {"op_F32_QASYMM8_SIGNED", REGISTER_FP32_NEON(fp32_i8_run_quantize_qasymm8)},
                        {"op_F32_QASYMM16", REGISTER_FP32_NEON(fp32_run_quantize_qasymm16)},

#ifdef BI_COMPUTE_ENABLE_FP16
                        {"op_F16_QASYMM8", REGISTER_FP16_NEON(fp16_u8_run_quantize_qasymm8)},
                        {"op_F16_QASYMM8_SIGNED", REGISTER_FP16_NEON(fp16_i8_run_quantize_qasymm8)},
                        {"op_F16_QASYMM16", REGISTER_FP16_NEON(fp16_run_quantize_qasymm16)},
#endif /* ARM_COMPUTE_ENABLE_FP16 */
                };

                std::string function_to_call("op_");

                // For offset only functions - must be 8-bit and have identical scale values.
                if (src->quantization_info().scale() == dst->quantization_info().scale() &&
                    (is_data_type_quantized_asymmetric_char(src->data_type()) &&
                     is_data_type_quantized_asymmetric_char(dst->data_type()))) {
                    function_to_call += "OFFSET_ONLY_";
                    // For optimized datatype conversion 8-bit re-quantization offset only functions.
                    // These must have an offset of exactly 128 to match requirements - has specific circumstances to match use case.
                    auto uqinfo =
                            compute_requantization_scale_offset(src->quantization_info().uniform(),
                                                                dst->quantization_info().uniform());
                    const auto src_dt = src->data_type();
                    if (src->data_type() != dst->data_type() &&
                        ((src_dt == BIDataType::QASYMM8_SIGNED && uqinfo.offset == 128) ||
                         (src_dt == BIDataType::QASYMM8 && uqinfo.offset == -128))) {
                        function_to_call += "CONVERT_";
                    }
                }

                // Specify datatype for function
                function_to_call += string_from_data_type(src->data_type()) + "_";
                function_to_call += string_from_data_type(dst->data_type());

                auto it = quant_map.find(function_to_call);

                if (it == quant_map.end()) {
                    BI_COMPUTE_ERROR("Unsupported combination of input and output data types");
                }
                _func = it->second;

                // Calculate window. Squash if possible.
                BIWindow win;
                std::tie(win, _split_dimension) = calculate_squashed_or_max_window(*src);

                BIICpuKernel::configure(win);
            }

            BIStatus BICpuQuantizeKernel::validate(const BIITensorInfo *src, const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
                return BIStatus{};
            }

            void BICpuQuantizeKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);
                BI_COMPUTE_ERROR_ON(_func == nullptr);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);
                (*_func)(src, dst, window);
            }

            const char *BICpuQuantizeKernel::name() const {
                return "BICpuQuantizeKernel";
            }

        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer