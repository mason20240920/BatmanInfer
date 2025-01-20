//
// Created by Mason on 2025/1/20.
//

#include <cpu/kernels/bi_cpu_convert_quantized_signedness_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments(const BIITensorInfo *src,
                                            const BIITensorInfo *dst) {
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1,
                                                                        BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED);

                    // 如果初始化验证输出合法
                    if (dst->total_size() != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1,
                                                                            BIDataType::QASYMM8,
                                                                            BIDataType::QASYMM8_SIGNED);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(src->tensor_shape(), dst->tensor_shape());
                    }

                    return BIStatus{};
                }

                std::pair<BIStatus, BIWindow> validate_and_configure_window(const BIITensorInfo *src,
                                                                            BIITensorInfo *dst) {
                    // 如果没有初始化则自动化`输出`同步
                    {
                        const bool is_input_signed = src->data_type() == BIDataType::QASYMM8_SIGNED;
                        const BIDataType dt = is_input_signed ? BIDataType::QASYMM8 : BIDataType::QASYMM8_SIGNED;
                        const BIUniformQuantizationInfo qinfo = src->quantization_info().uniform();
                        const int offset_correction = is_input_signed ? -128 : 128;
                        const BIQuantizationInfo corrected_qinfo = BIQuantizationInfo(qinfo.scale,
                                                                                      qinfo.offset + offset_correction);

                        auto_init_if_empty(*dst,
                                           src->clone()->set_data_type(dt).set_quantization_info(corrected_qinfo));
                    }

                    return std::make_pair(BIStatus{}, calculate_max_window(*dst));
                }
            } // namespace

            void BICpuConvertQuantizedSignednessKernel::configure(const BatmanInfer::BIITensorInfo *src,
                                                                  BatmanInfer::BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

                std::pair<BIStatus, BIWindow> win_config = validate_and_configure_window(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(win_config.first);
                BIICpuKernel::configure(win_config.second);
            }

            BIStatus BICpuConvertQuantizedSignednessKernel::validate(const BatmanInfer::BIITensorInfo *src,
                                                                     const BatmanInfer::BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
                return BIStatus{};
            }

            void BICpuConvertQuantizedSignednessKernel::run_op(BatmanInfer::BIITensorPack &tensors,
                                                               const BatmanInfer::BIWindow &window,
                                                               const BatmanInfer::ThreadInfo &info) {
                auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICPPKernel::window(), window);

                BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
                win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator input(src, win_collapsed);
                BIIterator output(src, win_collapsed);

                const int window_step_x = 16;
                const auto window_start_x = static_cast<int>(window.x().start());
                const auto window_end_x = static_cast<int>(window.x().end());

                const uint8_t mask = 128;
                const auto vmask = wrapper::vdup_n(mask, wrapper::traits::vector_128_tag{});

                execute_window_loop(
                        win_collapsed,
                        [&](const BICoordinates &) {
                            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            // Compute S elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const auto vin = wrapper::vloadq(input_ptr + x);
                                wrapper::vstore(output_ptr + x, wrapper::veor(vin, vmask));
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const uint8_t in = *(reinterpret_cast<const uint8_t *>(input_ptr + x));
                                *(output_ptr + x) = in ^ mask;
                            }
                        },
                        input, output);
            }

            const char *BICpuConvertQuantizedSignednessKernel::name() const {
                return "BICpuConvertQuantizedSignednessKernel";
            }
        } // namespace kernels
    }
}