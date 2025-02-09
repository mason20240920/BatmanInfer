//
// Created by Mason on 2025/2/9.
//

#include <cpu/kernels/bi_ne_normalization_layer_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <common/bi_registers.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_normalization_helpers.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <cpu/kernels/norm_layer/generic/neon/list.hpp>

namespace BatmanInfer {
    namespace {
        BIStatus validate_arguments(const BIITensorInfo *input,
                                    const BIITensorInfo *input_squared,
                                    const BIITensorInfo *output,
                                    const BINormalizationLayerInfo &norm_info) {
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, input_squared, output);
            BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, BIDataType::F16, BIDataType::F32);

            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, input_squared);
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, input_squared);
            BI_COMPUTE_RETURN_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");

            // Checks performed when output is configured
            if (output->total_size() != 0) {
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
            }

            return BIStatus{};
        }

    } // namespace

    BINENormalizationLayerKernel::BINENormalizationLayerKernel()
            : _func(nullptr), _input(nullptr), _input_squared(nullptr), _output(nullptr),
              _norm_info(BINormType::IN_MAP_1D) {
    }

    void BINENormalizationLayerKernel::configure(const BIITensor *input,
                                                 const BIITensor *input_squared,
                                                 BIITensor *output,
                                                 BINormalizationLayerInfo norm_info) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, input_squared, output);
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*output->info(), *input->info());

        // Perform validation step
        BI_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), input_squared->info(), output->info(), norm_info));

        const unsigned int norm_idx = get_normalization_dimension_index(input->info()->data_layout(), norm_info);

        _input = input;
        _input_squared = input_squared;
        _output = output;
        _norm_info = norm_info;
        switch (_input->info()->data_type()) {
            case BIDataType::F32: {
                switch (norm_idx) {
                    case 0: {
                        if (norm_info.type() == BINormType::IN_MAP_2D) {
                            _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_0_2D);
                        } else {
                            _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_0);
                        }
                        break;
                    }
                    case 1:
                        if (norm_info.type() == BINormType::IN_MAP_2D) {
                            _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_1_2D);
                        } else {
                            _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_1);
                        }
                        break;
                    case 2:
                        _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_2);
                        break;
                    default:
                        break;
                }
                break;
            }
#ifdef BI_COMPUTE_ENABLE_FP16
            case BIDataType::F16: {
                switch (norm_idx) {
                    case 0: {
                        if (norm_info.type() == BINormType::IN_MAP_2D) {
                            _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_0_2D);
                        } else {
                            _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_0);
                        }
                        break;
                    }
                    case 1:
                        if (norm_info.type() == BINormType::IN_MAP_2D) {
                            _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_1_2D);
                        } else {
                            _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_1);
                        }
                        break;
                    case 2:
                        _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_2);
                        break;
                    default:
                        break;
                }
                break;
            }
#endif /* BI_COMPUTE_ENABLE_FP16 */
            default:
                BI_COMPUTE_ERROR("NOT SUPPORTED!");
        }

        // Configure kernel window
        BIWindow win = calculate_max_window(*input->info(), BISteps());
        BIINEKernel::configure(win);
    }

    BIStatus BINENormalizationLayerKernel::validate(const BIITensorInfo *input,
                                                    const BIITensorInfo *input_squared,
                                                    const BIITensorInfo *output,
                                                    const BINormalizationLayerInfo norm_info) {
        BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, input_squared, output, norm_info));

        return BIStatus{};
    }

    void BINENormalizationLayerKernel::run(const BIWindow &window, const ThreadInfo &info) {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIINEKernel::window(), window);
        BI_COMPUTE_ERROR_ON(_func == nullptr);

        // Run function
        (*_func)(window, _input, _input_squared, _output, _norm_info);
    }
} // namespace BatmanInfer

