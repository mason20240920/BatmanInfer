//
// Created by Mason on 2025/1/18.
//

#include "cpu/kernels/bi_cpu_softmax_kernel.hpp"

#include "data/core/bi_error.h"
#include "data/core/bi_helpers.hpp"
#include "data/core/bi_i_tensor.hpp"
#include "data/core/bi_tensor_info.hpp"
#include "data/core/bi_utils.hpp"
#include "data/core/bi_window.hpp"

#include "common/bi_registers.hpp"
#include "data/core/cpp/bi_cpp_validate.hpp"
#include "data/core/helpers/bi_lut_manager.hpp"
#include "data/core/utils/helpers/bi_utils.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"
#include "cpu/kernels/softmax/list.hpp"
#include "data/core/helpers/bi_auto_configuration.hpp"

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {

                /* Softmax */
                static const std::vector<typename BICpuSoftmaxKernel::BISoftmaxKernel> available_kernels = {
                        {"neon_fp32_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (!data.is_log && data.dt == BIDataType::F32);
                                },
                                REGISTER_FP32_NEON(neon_fp32_softmax<false>)},
                        {"neon_fp16_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (!data.is_log && data.dt == BIDataType::F16) && data.isa.fp16;
                                },
                                REGISTER_FP16_NEON(neon_fp16_softmax<false>)},
                        {"neon_qu8_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (!data.is_log && data.dt == BIDataType::QASYMM8);
                                },
                                REGISTER_QASYMM8_NEON(neon_qasymm8_softmax<false>)},
                        {"neon_qs8_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (!data.is_log && data.dt == BIDataType::QASYMM8_SIGNED);
                                },
                                REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_softmax<false>)},
                        {"neon_fp32_log_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (data.is_log && data.dt == BIDataType::F32);
                                },
                                REGISTER_FP32_NEON(neon_fp32_softmax<true>)},
                        {"neon_fp16_log_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (data.is_log && data.dt == BIDataType::F16) && data.isa.fp16;
                                },
                                REGISTER_FP16_NEON(neon_fp16_softmax<true>)},
                        {"neon_qu8_log_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (data.is_log && data.dt == BIDataType::QASYMM8);
                                },
                                REGISTER_QASYMM8_NEON(neon_qasymm8_softmax<true>)},
                        {"neon_qs8_log_softmax",
                                [](const SoftmaxKernelDataTypeISASelectorData &data) {
                                    return (data.is_log && data.dt == BIDataType::QASYMM8_SIGNED);
                                },
                                REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_softmax<true>)},
                };

                BIStatus validate_arguments_softmax(
                        const BIITensorInfo &src, const BIITensorInfo &dst, float beta, int axis,
                        const BIITensorInfo &tmp,
                        bool is_log) {
                    BI_COMPUTE_UNUSED(beta);
                    // Check input
                    BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::F16, BIDataType::F32,
                                                                        BIDataType::BFLOAT16);

                    BI_COMPUTE_RETURN_ERROR_ON(axis < 0 || axis > 3);

                    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src.data_type());

                    // Check output if configured
                    if (dst.total_size() != 0) {
                        const BIQuantizationInfo output_quantization =
                                is_quantized_asymmetric ? BatmanInfer::get_softmax_output_quantization_info(
                                        src.data_type(), is_log)
                                                        : dst.quantization_info();
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);
                        BI_COMPUTE_RETURN_ERROR_ON(dst.quantization_info() != output_quantization);
                    }

                    // Check tmp if configured
                    if (tmp.total_size() != 0) {
                        // We have temporary storage only if src data type is quantized.
                        // Therefore, tmp data type must be F32
                        BI_COMPUTE_RETURN_ERROR_ON(tmp.data_type() != BIDataType::F32);
                        BI_COMPUTE_RETURN_ERROR_ON(!is_quantized_asymmetric);

                        // We could potentially reduce tmp memory if we could predict or make an assumption
                        // on the maximum number of threads that will run in parallel.
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &tmp);
                    }

                    return BIStatus{};
                }
            } // namespace

            const std::vector<typename BICpuSoftmaxKernel::BISoftmaxKernel> &
            BICpuSoftmaxKernel::get_available_kernels() {
                return available_kernels;
            }

            void BICpuSoftmaxKernel::configure(
                    const BIITensorInfo *src, BIITensorInfo *dst, float beta, bool is_log, int axis,
                    BIITensorInfo *tmp) {
                _axis = axis;

                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst, tmp);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments_softmax(*src, *dst, beta, axis, *tmp, is_log));

                // Configure kernel window
                const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src->data_type());

                // Output auto initialization if not yet initialized
                const BIQuantizationInfo output_quantization =
                        is_quantized_asymmetric ? BatmanInfer::get_softmax_output_quantization_info(src->data_type(),
                                                                                                    is_log)
                                                : dst->quantization_info();
                auto_init_if_empty(*dst, BITensorInfo(*src).set_quantization_info(output_quantization).reset_padding());

                // Tmp auto initialization if not yet initialized and src is quantized
                if (is_quantized_asymmetric) {
                    auto_init_if_empty(*tmp, BITensorInfo(*src).set_data_type(BIDataType::F32).reset_padding());
                }

                const auto *uk = BICpuSoftmaxKernel::get_implementation(SoftmaxKernelDataTypeISASelectorData{
                        src->data_type(), CPUInfo::get().get_isa(), is_log, axis,
                        CPUInfo::get().get_sme2_vector_length_in_bits()});
                BI_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

                std::string kernel_name = is_log ? std::string("BICpuLogSoftmaxKernel") : std::string(
                        "BICpuSoftmaxKernel");

                _beta = beta;
                _run_method = uk->ukernel;
                _name = kernel_name.append("/").append(uk->name);

                BIWindow win;

                int vec_size = 16 / dst->element_size();

                if (_axis == 0) {
                    win = calculate_max_window(*dst, BISteps());

                    /// TODO:Check dimensions > 0 for holes only. For this, we need
                    /// a utility function checking if there are holes after some dimension.
                    if (!has_holes(*dst, dst->num_dimensions() - 1)) {
                        win = win.collapse(win, BIWindow::DimY);
                    }
                } else if (_axis > 0 && _axis <= 3) {
                    win = calculate_max_window(*dst, BISteps(vec_size));
                } else {
                    BI_COMPUTE_ERROR("Invalid axis");
                }

                win.set(_axis, BIWindow::BIDimension(0, 1, 1));

                BIICpuKernel<BICpuSoftmaxKernel>::configure(win);

#ifdef __aarch64__
                const std::string uk_name = uk->name;

                if (src->data_type() == BIDataType::BFLOAT16) {
                    BILUTManager &lutmanager = BILUTManager::get_instance();
                    BILUTInfo info = {LUTType::Exponential, beta, BIDataType::BFLOAT16, BIUniformQuantizationInfo()};
                    _lut_bf16 = lutmanager.get_lut_table<LookupTable65536>(info);
                }

                if (uk_name == "sme2_qu8_softmax_lut_512VL" || uk_name == "sme2_qs8_softmax_lut_512VL") {
                    BIUniformQuantizationInfo qinfo = src->quantization_info().uniform();
                    // What the ukernel is interested in looking up is exp(b * deq(q)). The
                    // quantization offset cancels out in softmax so we don't need it in
                    // the LUT.
                    qinfo.offset = 0;
                    const BILUTInfo info{LUTType::Exponential, -beta, src->data_type(), qinfo};
                    _lut = BILUTManager::get_instance().get_lut_table<LookupTable256>(info);
                }
#endif // __aarch64__
            }

            BIStatus BICpuSoftmaxKernel::validate(
                    const BIITensorInfo *src, const BIITensorInfo *dst, float beta, int axis, bool is_log,
                    const BIITensorInfo *tmp) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst, tmp);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments_softmax(*src, *dst, beta, axis, *tmp, is_log));

                return BIStatus{};
            }

            void BICpuSoftmaxKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel<BICpuSoftmaxKernel>::window(), window);
                BI_COMPUTE_ERROR_ON(_run_method == nullptr);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST_0);

                if (is_data_type_quantized_asymmetric(src->info()->data_type())) {
                    auto tmp = tensors.get_tensor(BITensorType::ACL_DST_1);
                    unsigned int num_elems_processed_per_iteration;
                    if (_axis == 0) {
                        num_elems_processed_per_iteration = src->info()->valid_region().shape[_axis];
                    } else {
                        //16 QASYMM8/QASYMM8_SIGNED elements can fit into the 16-byte vectors.
                        num_elems_processed_per_iteration = 16;
                    }
                    const unsigned int tmp_size_for_thread =
                            tmp->info()->element_size() * num_elems_processed_per_iteration;

                    void *tmp_for_thread = tmp->buffer() + (info.thread_id * tmp_size_for_thread);
#ifdef __aarch64__
                    if (_lut) {
                        _run_method(src, tmp_for_thread, dst, _beta, _axis, window, _lut->data());
                    } else
#endif // __aarch64__
                    {
                        _run_method(src, tmp_for_thread, dst, _beta, _axis, window, nullptr);
                    }
                } else {
#ifdef __aarch64__
                    _run_method(src, nullptr, dst, _beta, _axis, window, _lut_bf16.get());
#else  // __aarch64__
                    _run_method(src, nullptr, dst, _beta, _axis, window, nullptr);
#endif // __aarch64__
                }
            }

            const char *BICpuSoftmaxKernel::name() const {
                return _name.c_str();
            }

        } // namespace kernels
    }
} // namespace BatmanInfer