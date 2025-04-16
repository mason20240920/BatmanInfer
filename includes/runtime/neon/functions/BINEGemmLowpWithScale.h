//
// Created by holynova on 25-4-12.
//

#pragma once

#include "runtime/bi_i_function.hpp"
#include "bi_ne_cast.h"
#include "bi_ne_gemm_lowp_matrix_mul_core.hpp"
#include "ne_pixel_wise_multiplication.hpp"
#include "runtime/bi_memory_group.hpp"
#include <runtime/bi_tensor.hpp>

namespace BatmanInfer {

    class BINEGemmLowpWithScale : public BIIFunction {
    public:
        explicit BINEGemmLowpWithScale(std::shared_ptr<BIIMemoryManager> memory_manager);
        BINEGemmLowpWithScale() : BINEGemmLowpWithScale(BIMemoryManagerOnDemand::make_default()) {}

        BINEGemmLowpWithScale(const BINEGemmLowpWithScale &) = delete;
        BINEGemmLowpWithScale &operator=(const BINEGemmLowpWithScale &) = delete;

        BINEGemmLowpWithScale(BINEGemmLowpWithScale &&) = delete;
        BINEGemmLowpWithScale &operator=(BINEGemmLowpWithScale &&) = delete;

        ~BINEGemmLowpWithScale() override;

        void configure(const BITensor *input,
                       const BITensor *gemm_weigth,
                       const BITensor *gemm_bias,
                       BITensor *output
                       );

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *gemm_weigth,
                                 const BIITensorInfo *output);

        void run() override;

    private:
        BINEGEMMLowpMatrixMultipleCore _gemm_lowp_layer;        // 做矩阵乘，但不考虑量化信息的 scale
        BINECast                       _cast_layer;             // 数据类型转换
        BINEPixelWiseMultiplication    _pixel_wise_mul_layer;   // 将量化信息的 scale 乘在矩阵乘的结果上

    private:
        BIMemoryGroup _memory_group;

        BITensor _gemm_lowp_output;
        BITensor _cast_output;
        BITensor _all_scales;

    };

} // namespace BatmanInfer
