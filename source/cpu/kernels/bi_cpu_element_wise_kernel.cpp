//
// Created by Mason on 2025/1/16.
//

#include "cpu/kernels/bi_cpu_element_wise_kernel.hpp"

#include "common/bi_registers.hpp"
#include "data/core/cpp/bi_cpp_validate.hpp"
#include "data/core/helpers/bi_auto_configuration.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"
#include "cpu/kernels/elementwise_binary/generic/list.hpp"

#include "neon/neon_defines.h"

#if defined(ENABLE_FP32_KERNELS)
namespace {
    static constexpr size_t default_min_max_mws_N1_fp32_neon = 25308;
    static constexpr size_t default_min_max_mws_V1_fp32_neon = 34772;
    static constexpr size_t default_div_mws_N1_fp32_neon = 19043;
    static constexpr size_t default_div_mws_V1_fp32_neon = 25511;
} // namespace
#endif /* ENABLE_FP32_KERNELS */

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                template<ArithmeticOperation op>
                const std::vector<BICpuElementwiseKernel<BICpuArithmeticKernel>::ElementwiseKernel> available_kernels_arithmetic = {
                        {"neon_fp32_arithmetic",

                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::F32 &&
                                           static_cast<ArithmeticOperation>(data.op) == op;
                                },
                                REGISTER_FP32_NEON(neon_fp32_elementwise_binary<op>)},
                        {"neon_s32_arithmetic",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::S32 &&
                                           static_cast<ArithmeticOperation>(data.op) == op;
                                },
                                REGISTER_INTEGER_NEON(neon_s32_elementwise_binary<op>)},
                        {"neon_fp16_arithmetic",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::F16 && data.isa.fp16 &&
                                           static_cast<ArithmeticOperation>(data.op) == op;
                                },
                                REGISTER_FP16_NEON(neon_fp16_elementwise_binary<op>)},
                        {"neon_s16_arithmetic",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::S16 &&
                                           static_cast<ArithmeticOperation>(data.op) == op;
                                },
                                REGISTER_INTEGER_NEON(neon_s16_elementwise_binary<op>)},
                        {"neon_qu8_arithmetic",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::QASYMM8 &&
                                           static_cast<ArithmeticOperation>(data.op) == op;
                                },
                                REGISTER_QASYMM8_NEON(neon_qasymm8_elementwise_binary<op>)},
                        {"neon_qs8_arithmetic",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::QASYMM8_SIGNED &&
                                           static_cast<ArithmeticOperation>(data.op) == op;
                                },
                                REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_elementwise_binary<op>)},
                };
                template<ComparisonOperation op>
                const std::vector<BICpuElementwiseKernel<BICpuComparisonKernel>::ElementwiseKernel> available_kernels_comperison = {
                        {"neon_u8_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::U8 && static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_INTEGER_NEON(neon_u8_comparison_elementwise_binary<op>)},
                        {"neon_fp32_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::F32 &&
                                           static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_FP32_NEON(neon_fp32_comparison_elementwise_binary<op>)},
                        {"neon_s16_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::S16 &&
                                           static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_INTEGER_NEON(neon_s16_comparison_elementwise_binary<op>)},
                        {"neon_s32_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::S32 &&
                                           static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_INTEGER_NEON(neon_s32_comparison_elementwise_binary<op>)},
                        {"neon_qu8_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::QASYMM8 &&
                                           static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_QASYMM8_NEON(neon_qasymm8_comparison_elementwise_binary<op>)},
                        {"neon_qs8_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::QASYMM8_SIGNED &&
                                           static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_comparison_elementwise_binary<op>)},
                        {"neon_fp16_comparison",
                                [](const ElementwiseDataTypeISASelectorData &data) {
                                    return data.dt == BIDataType::F16 && data.isa.fp16 &&
                                           static_cast<ComparisonOperation>(data.op) == op;
                                },
                                REGISTER_FP16_NEON(neon_fp16_comparison_elementwise_binary<op>)},
                };
            } // namespace

            const std::vector<BICpuElementwiseKernel<BICpuArithmeticKernel>::ElementwiseKernel> &
            BICpuArithmeticKernel::get_available_kernels() {
                static std::vector<BICpuElementwiseKernel<BICpuArithmeticKernel>::ElementwiseKernel> available_kernels;
                std::move(available_kernels_arithmetic<ArithmeticOperation::ADD>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::ADD>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::SUB>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::SUB>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::DIV>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::DIV>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::MIN>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::MIN>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::MAX>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::MAX>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::SQUARED_DIFF>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::SQUARED_DIFF>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::POWER>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::POWER>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_arithmetic<ArithmeticOperation::PRELU>.begin(),
                          available_kernels_arithmetic<ArithmeticOperation::PRELU>.end(),
                          std::back_inserter(available_kernels));

                return available_kernels;
            }

            const std::vector<BICpuElementwiseKernel<BICpuComparisonKernel>::ElementwiseKernel> &
            BICpuComparisonKernel::get_available_kernels() {
                static std::vector<BICpuElementwiseKernel<BICpuComparisonKernel>::ElementwiseKernel> available_kernels;
                std::move(available_kernels_comperison<ComparisonOperation::Equal>.begin(),
                          available_kernels_comperison<ComparisonOperation::Equal>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_comperison<ComparisonOperation::NotEqual>.begin(),
                          available_kernels_comperison<ComparisonOperation::NotEqual>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_comperison<ComparisonOperation::Greater>.begin(),
                          available_kernels_comperison<ComparisonOperation::Greater>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_comperison<ComparisonOperation::GreaterEqual>.begin(),
                          available_kernels_comperison<ComparisonOperation::GreaterEqual>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_comperison<ComparisonOperation::Less>.begin(),
                          available_kernels_comperison<ComparisonOperation::Less>.end(),
                          std::back_inserter(available_kernels));
                std::move(available_kernels_comperison<ComparisonOperation::LessEqual>.begin(),
                          available_kernels_comperison<ComparisonOperation::LessEqual>.end(),
                          std::back_inserter(available_kernels));

                return available_kernels;
            }

            template<class Derived>
            BIStatus BICpuElementwiseKernel<Derived>::validate_arguments_common(const BIITensorInfo &src0,
                                                                                const BIITensorInfo &src1,
                                                                                const BIITensorInfo &dst) {
                BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &src1);

                const BITensorShape out_shape = BITensorShape::broadcast_shape(src0.tensor_shape(),
                                                                               src1.tensor_shape());

                BI_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

                // Validate in case of configured dst
                if (dst.total_size() > 0) {
                    BI_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0),
                                                   "Wrong shape for output");
                }

                return BIStatus{};
            }

            void
            BICpuArithmeticKernel::configure_common(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                    BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

                const auto *uk = BICpuArithmeticKernel::get_implementation(
                        ElementwiseDataTypeISASelectorData{src0->data_type(), CPUInfo::get().get_isa(),
                                                           static_cast<int>(_op)});

                BI_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

                _run_method = uk->ukernel;
                _name = std::string("CpuArithmeticKernel").append("/").append(uk->name);

                // If any of shapes is dynamic, expect a configured window and dst at run-time.
                if (src0->is_dynamic() || src1->is_dynamic()) {
                    return;
                }

                auto shape_and_window = compute_output_shape_and_window(src0->tensor_shape(), src1->tensor_shape());
                auto_init_if_empty(*dst, shape_and_window.first, 1, src0->data_type());
                BIICpuKernel::configure(shape_and_window.second);
            }

            void
            BICpuComparisonKernel::configure_common(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                    BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

                const auto *uk = BICpuComparisonKernel::get_implementation(
                        ElementwiseDataTypeISASelectorData{src0->data_type(), CPUInfo::get().get_isa(),
                                                           static_cast<int>(_op)});

                BI_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

                _run_method = uk->ukernel;
                _name = std::string("CpuComparisonKernel").append("/").append(uk->name);

                // If any of shapes is dynamic, expect a configured window and dst at run-time.
                if (src0->is_dynamic() || src1->is_dynamic()) {
                    return;
                }

                auto shape_and_window = compute_output_shape_and_window(src0->tensor_shape(), src1->tensor_shape());
                auto_init_if_empty(*dst, shape_and_window.first, 1, src0->data_type());
                BIICpuKernel::configure(shape_and_window.second);
            }

            template<class Derived>
            void
            BICpuElementwiseKernel<Derived>::run_op(BIITensorPack &tensors, const BIWindow &window,
                                                    const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON(_run_method == nullptr);

                auto src0 = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                auto src1 = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                _run_method(src0, src1, dst, window);
            }

            template void
            BICpuElementwiseKernel<BICpuArithmeticKernel>::run_op(BIITensorPack &tensors, const BIWindow &window,
                                                                  const ThreadInfo &info);

            template void
            BICpuElementwiseKernel<BICpuComparisonKernel>::run_op(BIITensorPack &tensors, const BIWindow &window,
                                                                  const ThreadInfo &info);

            template<class Derived>
            const char *BICpuElementwiseKernel<Derived>::name() const {
                return _name.c_str();
            }

            template const char *BICpuElementwiseKernel<BICpuArithmeticKernel>::name() const;

            template const char *BICpuElementwiseKernel<BICpuComparisonKernel>::name() const;

/** Arithmetic operators (min, max, squared_diff) */
            void BICpuArithmeticKernel::configure(ArithmeticOperation op,
                                                  const BIITensorInfo *src0,
                                                  const BIITensorInfo *src1,
                                                  BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
                _op = op;
                BICpuArithmeticKernel::configure_common(src0, src1, dst);
            }

            BIStatus BICpuArithmeticKernel::validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1,
                                                               const BIITensorInfo &dst) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, BIDataType::QASYMM8,
                                                                    BIDataType::QASYMM8_SIGNED,
                                                                    BIDataType::S16, BIDataType::F16, BIDataType::S32,
                                                                    BIDataType::F32);
                // Validate in case of configured dst
                if (dst.total_size() > 0) {
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &dst);
                }
                return validate_arguments_common(src0, src1, dst);
            }

            BIStatus BICpuArithmeticKernel::validate(ArithmeticOperation op,
                                                     const BIITensorInfo *src0,
                                                     const BIITensorInfo *src1,
                                                     const BIITensorInfo *dst) {
                BI_COMPUTE_UNUSED(op);
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
                return BIStatus{};
            }

            size_t BICpuArithmeticKernel::get_mws(const CPUInfo &platform, size_t thread_count) const {
                BI_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
                if (this->_run_method == &neon_fp32_elementwise_binary<ArithmeticOperation::MIN> ||
                    this->_run_method == &neon_fp32_elementwise_binary<ArithmeticOperation::MAX>) {
                    size_t mws = BIICPPKernel::default_mws;
                    if (platform.get_cpu_model() == CPUModel::N1) {
                        mws = default_min_max_mws_N1_fp32_neon;
                    } else if (platform.get_cpu_model() == CPUModel::V1) {
                        mws = default_min_max_mws_V1_fp32_neon;
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
                ARM_COMPUTE_UNUSED(platform);
#endif /* ENABLE_FP32_KERNELS */
                return BIICPPKernel::default_mws;
            }

/** The division operator */

            void
            BICpuDivisionKernel::configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
                _op = ArithmeticOperation::DIV;
                BICpuArithmeticKernel::configure_common(src0, src1, dst);
            }

            size_t BICpuDivisionKernel::get_mws(const CPUInfo &platform, size_t thread_count) const {
                BI_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
                if (this->_run_method == &neon_fp32_elementwise_binary<ArithmeticOperation::DIV>) {
                    size_t mws = BIICPPKernel::default_mws;
                    if (platform.get_cpu_model() == CPUModel::N1) {
                        mws = default_div_mws_N1_fp32_neon;
                    } else if (platform.get_cpu_model() == CPUModel::V1) {
                        mws = default_div_mws_V1_fp32_neon;
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

            BIStatus BICpuDivisionKernel::validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1,
                                                             const BIITensorInfo &dst) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, BIDataType::S32, BIDataType::F16,
                                                                    BIDataType::F32);
                return BICpuArithmeticKernel::validate_arguments(src0, src1, dst);
            }

            BIStatus
            BICpuDivisionKernel::validate(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                          const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
                return BIStatus{};
            }

/** The power operator */
            void CpuPowerKernel::configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
                _op = ArithmeticOperation::POWER;
                BICpuArithmeticKernel::configure_common(src0, src1, dst);
            }

            BIStatus CpuPowerKernel::validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1,
                                                        const BIITensorInfo &dst) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, BIDataType::F16, BIDataType::F32);
                return BICpuArithmeticKernel::validate_arguments(src0, src1, dst);
            }

            BIStatus
            CpuPowerKernel::validate(const BIITensorInfo *src0, const BIITensorInfo *src1, const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
                return BIStatus{};
            }

/** Comparison operators (equal, not equal, less than, greater than, less than or equal, greater than or equal) */
            void BICpuComparisonKernel::configure(ComparisonOperation op,
                                                  const BIITensorInfo *src0,
                                                  const BIITensorInfo *src1,
                                                  BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
                _op = op;
                BICpuComparisonKernel::configure_common(src0, src1, dst);
            }

            BIStatus BICpuComparisonKernel::validate_arguments(const BIITensorInfo &src0, const BIITensorInfo &src1,
                                                               const BIITensorInfo &dst) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, BIDataType::U8, BIDataType::QASYMM8,
                                                                    BIDataType::QASYMM8_SIGNED, BIDataType::S16,
                                                                    BIDataType::F16,
                                                                    BIDataType::S32, BIDataType::F32);
                // Validate in case of configured dst
                if (dst.total_size() > 0) {
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, BIDataType::U8);
                }
                return validate_arguments_common(src0, src1, dst);
            }

            BIStatus BICpuComparisonKernel::validate(ComparisonOperation op,
                                                     const BIITensorInfo *src0,
                                                     const BIITensorInfo *src1,
                                                     const BIITensorInfo *dst) {
                BI_COMPUTE_UNUSED(op);
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
                return BIStatus{};
            }
        }
    }
}