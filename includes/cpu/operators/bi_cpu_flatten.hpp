//
// Created by Mason on 2025/1/22.
//

#pragma once

#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    class BIITensorInfo;

    namespace cpu {
        class BICpuReshape;

        /**
         * 把一个输入的算子进行平铺
         */
        class BICpuFlatten : public BIICpuOperator {
        public:
            BICpuFlatten();

            ~BICpuFlatten();

            /**
             * 支持数据类型: All
             * @param src
             * @param dst
             */
            void configure(const BIITensorInfo *src,
                           BIITensorInfo *dst);

            static BIStatus validate(const BIITensorInfo *src,
                                     const BIITensorInfo *dst);

            void run(BatmanInfer::BIITensorPack &tensors) override;

        private:
            std::unique_ptr<BICpuReshape> _reshape;
        };
    }
}