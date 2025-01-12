//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_CPU_ADD_HPP
#define BATMANINFER_BI_CPU_ADD_HPP

#include <function_info/bi_activationLayerInfo.h>
#include <cpu/bi_i_cpu_operator.hpp>
#include <data/core/bi_i_tensor_info.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * 基本的函数运行 @ref kernels::BICpuAddKernel
         */
        class BICpuAdd : public BIICpuOperator {
        public:
            /**
             * 初始化内核的输入、输出和边界模式:
             *
             * 有效的配置组合 (src0,src1) -> dst：
             *   - (U8,U8)           -> U8
             *   - (S16,S16)         -> S16
             *   - (S32,S32)         -> S32
             *   - (F16,F16)         -> F16
             *   - (F32,F32)         -> F32
             *   - (QASYMM8,QASYMM8) -> QASYMM8
             *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
             *   - (QSYMM16,QSYMM16) -> QSYMM16
             *
             * @param src0 第一个输入张量信息。支持的数据类型：U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
             * @param src1 第二个输入张量信息。支持的数据类型：U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
             * @param dst 输出张量信息。支持的数据类型：U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
             * @param policy 溢出策略
             * @param act_info （可选）融合激活层的信息。目前不支持。
             */
            void configure(const BIITensorInfo *src0,
                           const BIITensorInfo *src1,
                           BIITensorInfo *dst,
                           BIConvertPolicy policy,
                           const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            /**
             * 静态函数，用于检查给定的信息是否会产生有效的配置
             * @param src0
             * @param src1
             * @param dst
             * @param policy
             * @param act_info
             * @return
             */
            static BIStatus validate(const BIITensorInfo *src0,
                                     const BIITensorInfo *src1,
                                     const BIITensorInfo *dst,
                                     BIConvertPolicy policy,
                                     const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            // 重写
            void run(BatmanInfer::BIITensorPack &tensors) override;
        };
    }
}

#endif //BATMANINFER_BI_CPU_ADD_HPP
