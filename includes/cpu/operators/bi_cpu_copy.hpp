//
// Created by Mason on 2025/1/23.
//

#pragma once

#include "cpu/bi_i_cpu_operator.hpp"

namespace BatmanInfer {
    class BIITensorInfo;

    namespace cpu {
        /**
         * 基本函数: 运行@ref kernels::BICpuCopyKernel
         */
        class BICpuCopy : public BIICpuOperator {
        public:
            /**
             * 根据输入参数配置算子
             * @param src
             * @param dst
             */
            void configure(const BIITensorInfo *src, BIITensorInfo *dst);

            /**
             * @brief 动态配置
             * @param dst
             */
            void dynamic_configure(BIITensorInfo *dst);

            /**
             * 静态方法验证给定参数是否合法
             * @param src
             * @param dst
             * @return
             */
            static BIStatus validate(const BIITensorInfo *src,
                                     const BIITensorInfo *dst);
        };
    }
}
