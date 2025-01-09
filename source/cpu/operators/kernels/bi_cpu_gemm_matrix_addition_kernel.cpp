//
// Created by Mason on 2025/1/8.
//

#include <cpu/kernels/bi_cpu_gemm_matrix_addition_kernel.hpp>
#include <common/bi_registers.hpp>
#include <data/core/bi_helpers.hpp>

#include <cpu/kernels/gemm_matrix_add/list.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/cpp/bi_i_cpp_kernel.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                static const std::vector<BICpuGemmMatrixAdditionKernel::BIGemmMatrixAddKernel> available_kernels = {
                        {"neon_fp32_gemm_matrix_add",
                                [](const BIDataTypeISASelectorData &data) { return (data.dt == BIDataType::F32); },
                                REGISTER_FP32_NEON(neon_fp32_gemm_matrix_add)},
                        {"neon_fp16_gemm_matrix_add",
                                [](const BIDataTypeISASelectorData &data) {
                                    return (data.dt == BIDataType::F16) && data.isa.fp16;
                                },
                                REGISTER_FP16_NEON(neon_fp16_gemm_matrix_add)}
                };
            }  // namespace

            void BICpuGemmMatrixAdditionKernel::configure(const BIITensorInfo *src,
                                                          BIITensorInfo *dst, float beta) {
                BI_COMPUTE_UNUSED(dst);
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);

                // 运行步骤合法性
                BI_COMPUTE_ERROR_THROW_ON(BICpuGemmMatrixAdditionKernel::validate(src, dst, beta));

                _beta = beta;
                const auto uk = BICpuGemmMatrixAdditionKernel::get_implementation(
                        BIDataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa()});
                BI_COMPUTE_ERROR_ON_NULLPTR(uk);
                _func = uk->ukernel;
                // 配置内核窗口
                BIWindow win = calculate_max_window(*src, BISteps());
                BIICPPKernel::configure(win);
            }

            BIStatus BICpuGemmMatrixAdditionKernel::validate(const BIITensorInfo *src,
                                                             const BIITensorInfo *dst, float beta) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_UNUSED(beta);

                BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::F16, BIDataType::F32);

                if (dst->total_size() > 0) {
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                    BI_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(src, dst);
                }
                return BIStatus{};
            }

            void BICpuGemmMatrixAdditionKernel::run_op(BIITensorPack &tensors,
                                                       const BIWindow &window,
                                                       const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIIKernel::window(), window);
                BI_COMPUTE_ERROR_ON(tensors.empty());

                const BIITensor *src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                BIITensor *dst = tensors.get_tensor(BITensorType::ACL_DST);

                if (_beta != 0.0f)
                    (*_func)(src, dst, window, _beta);
            }

            const char *BICpuGemmMatrixAdditionKernel::name() const {
                return "BICpuGemmMatrixAdditionKernel";
            }

            const std::vector<BICpuGemmMatrixAdditionKernel::BIGemmMatrixAddKernel> &
            BICpuGemmMatrixAdditionKernel::get_available_kernels() {
                return available_kernels;
            }
        }
    }
}