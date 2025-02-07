//
// Created by Mason on 2025/2/7.
//

#include <cpu/operators/bi_cpu_gemm_lowp_output_stage.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_to_uint8_scale_by_fixed_point_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_to_int8_scale_by_fixed_point_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_to_int16_scale_by_fixed_point_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_scale_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuGemmLowpOutputStage::configure(BIITensorInfo *src,
                                                 BIITensorInfo *bias,
                                                 BIITensorInfo *dst,
                                                 const BIGEMMLowpOutputStageInfo &info) {
            // Perform validate step
            BI_COMPUTE_ERROR_THROW_ON(BICpuGemmLowpOutputStage::validate(src, bias, dst, info));
            BI_COMPUTE_LOG_PARAMS(src, bias, dst, info);

            switch (info.type) {
                case BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT: {
                    switch (info.output_data_type) {
                        case BIDataType::QASYMM8: {
                            auto k = std::make_unique<kernels::BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
                            k->configure(src, bias, dst, info.gemmlowp_multiplier, info.gemmlowp_shift,
                                         info.gemmlowp_offset,
                                         info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                            _kernel = std::move(k);
                            break;
                        }
                        case BIDataType::QASYMM8_SIGNED: {
                            auto k = std::make_unique<kernels::BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>();
                            k->configure(src, bias, dst, info.gemmlowp_multiplier, info.gemmlowp_shift,
                                         info.gemmlowp_offset,
                                         info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                            _kernel = std::move(k);
                            break;
                        }
                        case BIDataType::QSYMM16: {
                            auto k = std::make_unique<kernels::BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel>();
                            k->configure(src, bias, dst, info.gemmlowp_multiplier, info.gemmlowp_shift,
                                         info.gemmlowp_min_bound,
                                         info.gemmlowp_max_bound);
                            _kernel = std::move(k);
                            break;
                        }
                        default: {
                            BI_COMPUTE_ERROR("Unsupported output data type.");
                            break;
                        }
                    }
                    break;
                }
                case BIGEMMLowpOutputStageType::QUANTIZE_DOWN: {
                    switch (info.output_data_type) {
                        case BIDataType::QASYMM8:
                        case BIDataType::QASYMM8_SIGNED: {
                            auto k = std::make_unique<kernels::BICpuGemmLowpQuantizeDownInt32ScaleKernel>();
                            k->configure(src, bias, dst, &info);
                            _kernel = std::move(k);
                            break;
                        }
                        default: {
                            BI_COMPUTE_ERROR("Unsupported output data type.");
                            break;
                        }
                    }
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("Unsupported GEMMLowpOutputStage type.");
            }
        }

        BIStatus BICpuGemmLowpOutputStage::validate(const BIITensorInfo *src,
                                                    const BIITensorInfo *bias,
                                                    const BIITensorInfo *dst,
                                                    const BIGEMMLowpOutputStageInfo &info) {
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
            BI_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_type() == BIDataType::UNKNOWN,
                                           "BICpuGemmLowpOutputStage cannot be used with UNKNOWN output data type.");
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::QASYMM8, BIDataType::QASYMM8_SIGNED,
                                                                BIDataType::QSYMM16);
            BI_COMPUTE_RETURN_ERROR_ON((info.type != BIGEMMLowpOutputStageType::QUANTIZE_DOWN) &&
                                       (info.type != BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT));

            switch (info.type) {
                case BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT: {
                    switch (dst->data_type()) {
                        case BIDataType::QASYMM8:
                            return kernels::BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(
                                    src, bias, dst, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                        case BIDataType::QASYMM8_SIGNED:
                            return kernels::BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(
                                    src, bias, dst, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                        case BIDataType::QSYMM16:
                            return kernels::BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::validate(
                                    src, bias, dst, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                        default:
                            return BI_COMPUTE_CREATE_ERROR(BIErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
                    }
                }
                case BIGEMMLowpOutputStageType::QUANTIZE_DOWN: {
                    switch (dst->data_type()) {
                        case BIDataType::QASYMM8:
                        case BIDataType::QASYMM8_SIGNED:
                            return kernels::BICpuGemmLowpQuantizeDownInt32ScaleKernel::validate(src, bias, dst, &info);
                        default:
                            return BI_COMPUTE_CREATE_ERROR(BIErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
                    }
                }
                default:
                    return BI_COMPUTE_CREATE_ERROR(BIErrorCode::RUNTIME_ERROR, "Unsupported GEMMLowpOutputStage type.");
            }
        }

        void BICpuGemmLowpOutputStage::run(BIITensorPack &tensors) {
            BINEScheduler::get().schedule_op(_kernel.get(), BIWindow::DimY, _kernel->window(), tensors);
        }
    } // namespace cpu
} // namespace BatmanInfer