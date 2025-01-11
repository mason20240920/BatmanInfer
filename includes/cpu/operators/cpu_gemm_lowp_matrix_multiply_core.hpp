//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_CPU_GEMM_LOWP_MATRIX_MULTIPLY_CORE_HPP
#define BATMANINFER_CPU_GEMM_LOWP_MATRIX_MULTIPLY_CORE_HPP

#include <data/core/bi_i_tensor_info.hpp>
#include <function_info/bi_GEMMInfo.h>

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/bi_i_cpu_operator.hpp>

#include <memory>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            class BICpuGemmInterleave4x4Kernel;

            class BICpuGemmLowpMatrixMultiplyKernel;

            class BICpuGemmLowpOffsetContributionKernel;

            class BICpuGemmLowpOffsetContributionOutputStageKernel;

            class BICpuGemmLowpMatrixAReductionKernel;

            class BICpuGemmLowpMatrixBReductionKernel;

            class BICpuGemmTranspose1xWKernel;

            class BICpuConvertQuantizedSignednessKernel;
        } // namespace kernels
        class BICpuGemmAssemblyDispatch;

        class BICpuActivation;

        /**
         * @brief
         */
        class BICpuGemmLowpMatrixMultiplyCore : public BIICpuOperator {

        private:
            enum AuxTensorIdx {
                /* 枚举值的 0 到 2 号索引被保留给 BICpuGemmAssemblyDispatch */
                VectorSumCol = 3,
                VectorSumRow,
                TmpA,
                TmpB,
                MMResultS32,
                SignedA,
                SignedOutput,
                Count
            };
        };
    }
}

#endif //BATMANINFER_CPU_GEMM_LOWP_MATRIX_MULTIPLY_CORE_HPP
