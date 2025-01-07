//
// Created by Mason on 2025/1/6.
//

#ifndef BATMANINFER_CPU_TRANSPOSE_HPP
#define BATMANINFER_CPU_TRANSPOSE_HPP

#include <cpu/bi_i_cpu_operator.hpp>
#include <data/core/bi_i_tensor_info.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * @brief 基本函数以运行 @ref kernels::BICpuTransposeKernel
         */
        class BICpuTranspose : public BIICpuOperator {
        public:
            /**
             * @brief 根据给定的参数配置算子
             * @param src 源张量进行置换。支持的数据类型：全部。
             * @param dst 目标张量, 数据类型支持: @p src 一致
             */
            void configure(const BIITensorInfo *src,
                           BIITensorInfo *dst);

            /**
             * @brief 静态方法确认给定的参数是否是有效配置
             * @param src 与 @ref BICpuTranspose::configure() 一致
             * @param dst 与 @ref BICpuTranspose::configure() 一致
             * @return 返回配置状态
             */
            static BIStatus validate(const BIITensorInfo *src,
                                     const BIITensorInfo *dst);
        };
    }
}

#endif //BATMANINFER_CPU_TRANSPOSE_HPP
