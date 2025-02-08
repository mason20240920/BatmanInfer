//
// Created by Mason on 2025/2/8.
//

#include <cpu/kernels/bi_cpu_dequantize_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <common/bi_registers.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <cpu/kernels/dequantize/generic/neon/list.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments(const BIITensorInfo *src, const BIITensorInfo *dst) {
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::QSYMM8_PER_CHANNEL,
                                                                        BIDataType::QSYMM8,
                                                                        BIDataType::QSYMM16);

                    if (dst->tensor_shape().total_size() > 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(dst);
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::F16, BIDataType::F32);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
                    }

                    return BIStatus{};
                }
            } // namespace

            void
            BICpuDequantizeKernel::configure(const BatmanInfer::BIITensorInfo *src, BatmanInfer::BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

                // Configure kernel window
                BIWindow win = calculate_max_window(*src, BISteps());

                // Output tensor auto initialization if not yet initialized
                auto_init_if_empty(*dst, src->tensor_shape(), 1, BIDataType::F32);

                BIICpuKernel::configure(win);

                switch (dst->data_type()) {
                    case BIDataType::F32:
                        _func = REGISTER_FP32_NEON(fp32_run_dequantization_core);
                        break;
#ifdef BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F16:
                        _func = REGISTER_FP16_NEON(fp16_run_dequantization_core);
                        break;
#endif /* BI_COMPUTE_ENABLE_FP16 */
                    default:
                        BI_COMPUTE_ERROR("Unsupported data type.");
                }
            }

            BIStatus BICpuDequantizeKernel::validate(const BIITensorInfo *src, const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
                return BIStatus{};
            }

            void BICpuDequantizeKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);
                _func(src, dst, window);
            }

            const char *BICpuDequantizeKernel::name() const {
                return "BICpuDequantizeKernel";
            }
        } // namespace kernels
    }
}