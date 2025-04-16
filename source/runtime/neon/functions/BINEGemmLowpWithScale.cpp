//
// Created by holynova on 25-4-15.
//

#include "runtime/neon/functions/BINEGemmLowpWithScale.h"
#include "common/utils/bi_log.hpp"

namespace BatmanInfer {

    BINEGemmLowpWithScale::~BINEGemmLowpWithScale() = default;

    BINEGemmLowpWithScale::BINEGemmLowpWithScale(std::shared_ptr<BIIMemoryManager> memory_manager) :
        _gemm_lowp_layer(),
        _cast_layer(),
        _pixel_wise_mul_layer(),
        _memory_group(std::move(memory_manager)),
        _gemm_lowp_output(),
        _cast_output(),
        _all_scales()
    { }

    BIStatus BINEGemmLowpWithScale::validate(const BIITensorInfo *input,
                                             const BIITensorInfo *gemm_weigth,
                                             const BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, gemm_weigth, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::QASYMM8);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(gemm_weigth, BIDataType::QSYMM8_PER_CHANNEL);

        return BIStatus{};
    }

    void BINEGemmLowpWithScale::configure(const BITensor *input,
                                          const BITensor *gemm_weigth,
                                          const BITensor *gemm_bias,
                                          BITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, gemm_weigth, output);
        BI_COMPUTE_ERROR_THROW_ON(
            BINEGemmLowpWithScale::validate(input->info(), gemm_weigth->info(), output->info()));

        auto output_shape = BITensorShape(output->info()->tensor_shape());

        BIQuantizationInfo gemm_qinfo = gemm_weigth->info()->quantization_info();
        const auto scale_shape = BITensorShape(gemm_qinfo.scale().size(), 1);

        _gemm_lowp_output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::S32));
        _cast_output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
        _all_scales.allocator()->init(BITensorInfo(scale_shape, 1, BIDataType::F16));

        _memory_group.manage(&_gemm_lowp_output);
        _memory_group.manage(&_cast_output);
        // _memory_group.manage(&_all_scales);

        _gemm_lowp_output.allocator()->allocate();
        _cast_output.allocator()->allocate();
        _all_scales.allocator()->allocate();

        GEMMInfo gemm_info = GEMMInfo(false,
                                      false,
                                      true,
                                      false,
                                      false,
                                      false,
                                      BIGEMMLowpOutputStageInfo(),
                                      false, true, false,
                                      BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
        _gemm_lowp_layer.configure(input, gemm_weigth, gemm_bias, &_gemm_lowp_output, gemm_info);
        _cast_layer.configure(&_gemm_lowp_output, &_cast_output, BIConvertPolicy::WRAP);

        auto scale_ptr = reinterpret_cast<float16_t*>(_all_scales.buffer());
        const float input_scale = input->info()->quantization_info().scale()[0];
        for (int i = 0; i < scale_shape[BIWindow::DimY]; ++i) {
            for (int j = 0; j < scale_shape[BIWindow::DimX]; ++j) {
                scale_ptr[i * scale_shape[BIWindow::DimX] + j] =
                    static_cast<float16_t>(input_scale * gemm_qinfo.scale()[i * scale_shape[BIWindow::DimX] + j]);
            }
        }

        _pixel_wise_mul_layer.configure(&_cast_output, &_all_scales, output, 1.0f,
            BIConvertPolicy::WRAP, BIRoundingPolicy::TO_ZERO);

    }

    void BINEGemmLowpWithScale::run() {
        BIMemoryGroupResourceScope scope_mg(_memory_group);

        _gemm_lowp_layer.run();
        _cast_layer.run();
        _pixel_wise_mul_layer.run();
    }

} // namespace BatmanInfer
