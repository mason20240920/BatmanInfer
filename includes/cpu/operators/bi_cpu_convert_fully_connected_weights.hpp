//
// Created by Mason on 2025/1/22.
//

#pragma once

#include <cpu/bi_i_cpu_operator.hpp>
#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    class BIITensorInfo;
    namespace cpu {
        /**
         * 基础算子: 运行@ref kernels::BICpuConvertFullyConnectedWeightsKernel
         */
        class BICpuConvertFullyConnectedWeights : public BIICpuOperator {
        public:
            void configure(const BIITensorInfo *src,
                           BIITensorInfo *dst,
                           const BITensorShape &original_src_shape,
                           BIDataLayout data_layout);

            static BIStatus validate(const BIITensorInfo *src,
                                     const BIITensorInfo *dst,
                                     const BITensorShape &original_src_shape,
                                     BIDataLayout data_layout);

            void run(BatmanInfer::BIITensorPack &tensors) override;
        };
    }
}