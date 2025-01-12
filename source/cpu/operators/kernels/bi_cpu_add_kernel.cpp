//
// Created by Mason on 2025/1/12.
//

#include <cpu/kernels/bi_cpu_add_kernel.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_vlidate.hpp>

#include <common/bi_registers.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <cpu/kernels/add/list.hpp>

#include <array>

#if defined(ENABLE_FP32_KERNELS)
namespace {
    static constexpr size_t default_mws_N1_fp32_neon = 24536;
    static constexpr size_t default_mws_V1_fp32_neon = 40510;
} // namespace
#endif /* ENABLE_FP32_KERNELS */

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                static const std::vector<BICpuAddKernel::BIAddKernel> available_kernels = {
                        {"neon_qu8_add_fixedpoint",
                                          [](const CpuAddKernelDataTypeISASelectorData &data) {
                                              return (data.dt == BIDataType::QASYMM8) && data.can_use_fixedpoint;
                                          },
                                REGISTER_FP32_NEON(BatmanInfer::cpu::add_q8_neon_fixedpoint<uint8_t>)},
                        {"neon_qs8_add_fixedpoint",
                                          [](const CpuAddKernelDataTypeISASelectorData &data) {
                                              return (data.dt == BIDataType::QASYMM8_SIGNED) && data.can_use_fixedpoint;
                                          },
                                REGISTER_FP32_NEON(BatmanInfer::cpu::add_q8_neon_fixedpoint<int8_t>)},
                        {"sve2_qu8_add",
                                          [](const CpuAddKernelDataTypeISASelectorData &data) {
                                              return (data.dt == BIDataType::QASYMM8) && data.isa.sve2;
                                          },
                                REGISTER_FP32_NEON(BatmanInfer::cpu::add_fp32_neon)},
                        {"neon_fp16_add",
                                          [](const CpuAddKernelDataTypeISASelectorData &data) {
                                              return (data.dt == BIDataType::F16) && data.isa.fp16;
                                          },
                                REGISTER_FP16_NEON(BatmanInfer::cpu::add_fp16_neon)},
                        {"neon_u8_add",   [](const CpuAddKernelDataTypeISASelectorData &data) {
                            return (data.dt == BIDataType::U8);
                        },
                                REGISTER_INTEGER_NEON(BatmanInfer::cpu::add_u8_neon)},
                        {"neon_s16_add",  [](const CpuAddKernelDataTypeISASelectorData &data) {
                            return (data.dt == BIDataType::S16);
                        },
                                REGISTER_INTEGER_NEON(BatmanInfer::cpu::add_s16_neon)},
                        {"neon_s32_add",  [](const CpuAddKernelDataTypeISASelectorData &data) {
                            return (data.dt == BIDataType::S32);
                        },
                                REGISTER_INTEGER_NEON(BatmanInfer::cpu::add_s32_neon)},
                        {"neon_qu8_add",  [](const CpuAddKernelDataTypeISASelectorData &data) {
                            return (data.dt == BIDataType::QASYMM8);
                        },
                                REGISTER_QASYMM8_NEON(BatmanInfer::cpu::add_qasymm8_neon)},
                        {"neon_qs8_add",
                                          [](const CpuAddKernelDataTypeISASelectorData &data) {
                                              return (data.dt == BIDataType::QASYMM8_SIGNED);
                                          },
                                REGISTER_QASYMM8_SIGNED_NEON(BatmanInfer::cpu::add_qasymm8_signed_neon)},
                        {"neon_qs16_add", [](const CpuAddKernelDataTypeISASelectorData &data) {
                            return (data.dt == BIDataType::QSYMM16);
                        },
                                REGISTER_QSYMM16_NEON(BatmanInfer::cpu::add_qsymm16_neon)}
                };

                BIStatus
                validate_arguments(const BIITensorInfo &src0,
                                   const BIITensorInfo &src1,
                                   const BIITensorInfo &dst,
                                   BIConvertPolicy policy) {
                    BI_COMPUTE_UNUSED(policy);

                    BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1,
                                                                        BIDataType::U8,
                                                                        BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::S16,
                                                                        BIDataType::QSYMM16,
                                                                        BIDataType::F16,
                                                                        BIDataType::S32,
                                                                        BIDataType::F32);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &src1);

                    const BITensorShape out_shape = BITensorShape::broadcast_shape(src0.tensor_shape(),
                                                                                   src1.tensor_shape());

                    BI_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
                    BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            (src0.tensor_shape().x() != src1.tensor_shape().x()) &&
                            ((src0.data_type() != src1.data_type()) || (src0.data_type() != dst.data_type()) ||
                             (src1.data_type() != dst.data_type())),
                            "Broadcasting across width is supported on configurations where all tensors have the same data type");

                    // Validate in case of configured dst
                    if (dst.total_size() > 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &dst);
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0),
                                "Wrong shape for dst");
                    }

                    const auto can_use_fixedpoint = add_q8_neon_fixedpoint_possible(&src0, &src1, &dst);
#ifdef ARM_COMPUTE_ENABLE_SME2
                    const auto can_use_sme2_impl = add_q8_sme2_fixedpoint_possible(&src0, &src1, &dst);
#else  /* ARM_COMPUTE_ENABLE_SME2 */
                    const auto can_use_sme2_impl = false;
#endif /* ARM_COMPUTE_ENABLE_SME2 */
                    const auto uk =
                            BICpuAddKernel::get_implementation<CpuAddKernelDataTypeISASelectorData>(
                                    CpuAddKernelDataTypeISASelectorData{
                                            src0.data_type(), CPUInfo::get().get_isa(), can_use_fixedpoint,
                                            can_use_sme2_impl});

                    BI_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

                    return BIStatus{};
                }
            } // namespace

            void
            BICpuAddKernel::configure(const BatmanInfer::BIITensorInfo *src0,
                                      const BatmanInfer::BIITensorInfo *src1,
                                      BatmanInfer::BIITensorInfo *dst,
                                      BatmanInfer::BIConvertPolicy policy) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst, policy));

                const auto can_use_fixedpoint = add_q8_neon_fixedpoint_possible(src0, src1, dst);
#ifdef ARM_COMPUTE_ENABLE_SME2
                const auto can_use_sme2_impl = add_q8_sme2_fixedpoint_possible(src0, src1, dst);
#else  /* ARM_COMPUTE_ENABLE_SME2 */
                const auto can_use_sme2_impl = false;
#endif /* ARM_COMPUTE_ENABLE_SME2 */
                const auto uk =
                        BICpuAddKernel::get_implementation<CpuAddKernelDataTypeISASelectorData>(
                                CpuAddKernelDataTypeISASelectorData{
                                        src0->data_type(), CPUInfo::get().get_isa(), can_use_fixedpoint,
                                        can_use_sme2_impl});

                BI_COMPUTE_ERROR_ON_NULLPTR(uk);

                _policy = policy;
                _run_method = uk->ukernel;
                _name = std::string("CpuAddKernel").append("/").append(uk->name);

                // Auto initialize dst if not initialized
                const BITensorShape &out_shape = BITensorShape::broadcast_shape(src0->tensor_shape(),
                                                                                src1->tensor_shape());
                set_shape_if_empty(*dst, out_shape);
                set_data_type_if_unknown(*dst, src0->data_type());

                // Configure kernel window
                BIWindow win;
                std::tie(win, _split_dimension) = calculate_squashed_or_max_window(*src0, *src1);

                BIICpuKernel::configure(win);
            }

            BIStatus
            BICpuAddKernel::validate(const BatmanInfer::BIITensorInfo *src0, const BatmanInfer::BIITensorInfo *src1,
                                     const BatmanInfer::BIITensorInfo *dst, BatmanInfer::BIConvertPolicy policy) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);

                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst, policy));

                return BIStatus{};
            }

            void BICpuAddKernel::run_op(BatmanInfer::BIITensorPack &tensors, const BatmanInfer::BIWindow &window,
                                        const BatmanInfer::ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                BI_COMPUTE_ERROR_ON(tensors.empty());
                BI_COMPUTE_ERROR_ON(_run_method == nullptr);

                const BIITensor *src0 = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                const BIITensor *src1 = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                BIITensor *dst = tensors.get_tensor(BITensorType::ACL_DST);
                _run_method(src0, src1, dst, _policy, window);
            }

            const char *BICpuAddKernel::name() const {
                return _name.c_str();
            }

            const std::vector<BICpuAddKernel::BIAddKernel> &BICpuAddKernel::get_available_kernels() {
                return available_kernels;
            }

            size_t BICpuAddKernel::get_mws(const BatmanInfer::CPUInfo &platform, size_t thread_count) const {
                BI_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
                if (this->_run_method == &add_fp32_neon) {
                    size_t mws = BIICPPKernel::default_mws;
                    if (platform.get_cpu_model() == CPUModel::N1) {
                        mws = default_mws_N1_fp32_neon;
                    } else if (platform.get_cpu_model() == CPUModel::V1) {
                        mws = default_mws_V1_fp32_neon;
                    } else {
                        return BIICPPKernel::default_mws;
                    }

                    // tensor is 1D or was re-interpreted as 1D
                    if (this->window().shape().num_dimensions() == 1) {
                        return mws;
                    } else {
                        // scale mws down by the number of elements along all the dimensions (x, z, w, etc) except the one
                        // that we parallelize along (the y dimension). This allows for parallelization when the Y_SIZE is small
                        // but the other sizes are large, which boosts performance.
                        mws = static_cast<size_t>(mws / (this->window().num_iterations_total() /
                                                         this->window().num_iterations(1)));
                        return std::max(static_cast<size_t>(1), mws);
                    }
                }
#else  /* ENABLE_FP32_KERNELS */
                BI_COMPUTE_UNUSED(platform);
#endif /* ENABLE_FP32_KERNELS */
                return BIICPPKernel::default_mws;
            }
        } // namespace kernels
    }
}