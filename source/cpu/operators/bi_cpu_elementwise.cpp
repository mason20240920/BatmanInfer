//
// Created by Mason on 2025/1/16.
//

#include <cpu/operators/bi_cpu_elementwise.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <cpu/kernels/bi_cpu_element_wise_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuElementwiseBase::run(BIITensorPack &tensors) {
            // If the kernel has been configured, use the window from the kernel.
            if (_kernel->is_window_configured()) {
                BIICpuOperator::run(tensors);
                return;
            }

            auto src0_info = tensors.get_const_tensor(BITensorType::ACL_SRC_0)->info();
            auto src1_info = tensors.get_const_tensor(BITensorType::ACL_SRC_1)->info();
            auto shape_and_window = compute_output_shape_and_window(src0_info->tensor_shape(),
                                                                    src1_info->tensor_shape());
            BIICpuOperator::run(tensors, shape_and_window.second);
        }

        template<ArithmeticOperation op>
        void
        BICpuElementwiseArithmetic<op>::configure(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                  BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src0, src1, dst);
            auto k = std::make_unique<kernels::BICpuArithmeticKernel>();
            k->configure(op, src0, src1, dst);
            _kernel = std::move(k);
        }

        template<ArithmeticOperation op>
        BIStatus BICpuElementwiseArithmetic<op>::validate(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                          const BIITensorInfo *dst) {
            return kernels::BICpuArithmeticKernel::validate(op, src0, src1, dst);
        }

        template
        class BICpuElementwiseArithmetic<ArithmeticOperation::MAX>;

        template
        class BICpuElementwiseArithmetic<ArithmeticOperation::MIN>;

        template
        class BICpuElementwiseArithmetic<ArithmeticOperation::SQUARED_DIFF>;

        template
        class BICpuElementwiseArithmetic<ArithmeticOperation::PRELU>;

        void
        BICpuElementwiseDivision::configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src0, src1, dst);
            auto k = std::make_unique<kernels::BICpuDivisionKernel>();
            k->configure(src0, src1, dst);
            _kernel = std::move(k);
        }

        BIStatus
        BICpuElementwiseDivision::validate(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                           const BIITensorInfo *dst) {
            return kernels::BICpuDivisionKernel::validate(src0, src1, dst);
        }

        void
        BICpuElementwisePower::configure(const BIITensorInfo *src0, const BIITensorInfo *src1, BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src0, src1, dst);
            auto k = std::make_unique<kernels::CpuPowerKernel>();
            k->configure(src0, src1, dst);
            _kernel = std::move(k);
        }

        BIStatus
        BICpuElementwisePower::validate(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                        const BIITensorInfo *dst) {
            return kernels::CpuPowerKernel::validate(src0, src1, dst);
        }

        template<ComparisonOperation COP>
        void BICpuElementwiseComparisonStatic<COP>::configure(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                              BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src0, src1, dst);
            auto k = std::make_unique<kernels::BICpuComparisonKernel>();
            k->configure(COP, src0, src1, dst);
            _kernel = std::move(k);
        }

        template<ComparisonOperation COP>
        BIStatus
        BICpuElementwiseComparisonStatic<COP>::validate(const BIITensorInfo *src0, const BIITensorInfo *src1,
                                                        const BIITensorInfo *dst) {
            return kernels::BICpuComparisonKernel::validate(COP, src0, src1, dst);
        }

        void BICpuElementwiseComparison::configure(const BIITensorInfo *src0,
                                                   const BIITensorInfo *src1,
                                                   BIITensorInfo *dst,
                                                   ComparisonOperation op) {
            BI_COMPUTE_LOG_PARAMS(src0, src1, dst);
            auto k = std::make_unique<kernels::BICpuComparisonKernel>();
            k->configure(op, src0, src1, dst);
            _kernel = std::move(k);
        }

        BIStatus BICpuElementwiseComparison::validate(const BIITensorInfo *src0,
                                                      const BIITensorInfo *src1,
                                                      const BIITensorInfo *dst,
                                                      ComparisonOperation op) {
            return kernels::BICpuComparisonKernel::validate(op, src0, src1, dst);
        }

// Supported Specializations
        template
        class BICpuElementwiseComparisonStatic<ComparisonOperation::Equal>;

        template
        class BICpuElementwiseComparisonStatic<ComparisonOperation::NotEqual>;

        template
        class BICpuElementwiseComparisonStatic<ComparisonOperation::Greater>;

        template
        class BICpuElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;

        template
        class BICpuElementwiseComparisonStatic<ComparisonOperation::Less>;

        template
        class BICpuElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
    }
}