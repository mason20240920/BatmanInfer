//
// Created by Mason on 2025/2/12.
//
#include <cpu/kernels/BINERMSNormLayerKernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <common/bi_registers.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <cpu/kernels/rms_norm/generic/neon/list.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace {
            BIStatus validate_arguments(const BIITensorInfo *input,
                                        const BIITensorInfo *gamma,
                                        const BIITensorInfo *output) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, gamma, output);
                BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, BIDataType::F16);

                BI_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, gamma, output);
                // 检查 H 维度匹配
                BI_COMPUTE_ERROR_ON(input->dimension(0) != gamma->dimension(0));
                // 检查是否能被8整除
                BI_COMPUTE_ERROR_ON_MSG(gamma->dimension(0) % 8 != 0, "HIDDEN SIZE MUST BE DIVIDED BY 8!");

                // Checks performed when output is configured
                if (output->total_size() != 0) {
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
                }

                return BIStatus{};
            }
        } // namespace

        BINERMSNormLayerKernel::BINERMSNormLayerKernel() : _func(nullptr), _input(nullptr), _gamma(nullptr),
                                                           _output(nullptr) {}

        void BINERMSNormLayerKernel::configure(const BatmanInfer::BIITensor *input, const BatmanInfer::BIITensor *gamma,
                                               BatmanInfer::BIITensor *output) {
            BI_COMPUTE_ERROR_ON_NULLPTR(input, gamma, output);
            // Output tensor auto initialization if not yet initialized
            auto_init_if_empty(*output->info(), *input->info());

            // Perform validation step
            BI_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), gamma->info(), output->info()));

            _input = input;
            _gamma = gamma;
            _output = output;
            switch (_input->info()->data_type()) {
#ifdef BI_COMPUTE_ENABLE_FP16
                case BIDataType::F16: {
                    _func = REGISTER_FP32_NEON(cpu::neon_rms_norm_fp16_8_impl);
                    break;
                }
#endif
                default:
                    BI_COMPUTE_ERROR("NOT SUPPORTED");
            }

            // Configure kernel window
            BIWindow win = calculate_max_window(*input->info(), BISteps());
            BIINEKernel::configure(win);
        }

        BIStatus BINERMSNormLayerKernel::validate(const BatmanInfer::BIITensorInfo *input,
                                                  const BatmanInfer::BIITensorInfo *gamma,
                                                  const BatmanInfer::BIITensorInfo *output) {
            BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, gamma, output));

            return BIStatus{};
        }

        void BINERMSNormLayerKernel::run(const BatmanInfer::BIWindow &window, const BatmanInfer::ThreadInfo &info) {
            BI_COMPUTE_UNUSED(info);
            BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
            BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIINEKernel::window(), window);
            BI_COMPUTE_ERROR_ON(_func == nullptr);
            // Run function
            (*_func)(window, _input, _gamma, _output);
        }

        bool BINERMSNormLayerKernel::is_parallelisable() const {
            return false;
        }
    }
}