//
// Created by Mason on 2025/1/13.
//

#ifndef BATMANINFER_BI_ASSEMBLY_UTILS_HPP
#define BATMANINFER_BI_ASSEMBLY_UTILS_HPP

#include <data/core/bi_types.hpp>

#include <data/core/neon/kernels/assembly/common.hpp>
#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

namespace BatmanInfer {
    class BIActivationLayerInfo;

    namespace assembly_utils {
        /**
         * @brief 执行 Compute Library ActivationLayerInfo 与程序集 Activation 结构之间的映射。
         *
         * @param[in] act 计算库的激活函数信息
         * @return 汇编激活函数信息
         */
        BatmanGemm::Activation map_to_batman_gemm_activation(const BIActivationLayerInfo &act);

        /**
         * @brief 执行 Compute Library PadStrideInfo 和汇编 PaddingValues 结构之间的映射。
         * @param pad_stride_info
         * @return
         */
        BatmanConv::PaddingValues map_to_batman_conv_padding(const BIPadStrideInfo &pad_stride_info);

        /**
         * 执行从 Compute Library WeightFormat 到程序集 WeightFormat 枚举的映射。
         *
         * @param weight_format 计算库的权重枚举类型
         * @return  汇编权重格式
         */
        BatmanGemm::WeightFormat map_to_batman_gemm_weight_format(const BatmanInfer::BIWeightFormat &weight_format);

        /**
         * 与上面相反
         * @param weight_format
         * @return
         */
        BatmanInfer::BIWeightFormat map_to_batman_compute_weight_format(const BatmanGemm::WeightFormat &weight_format);
    }
}


#endif //BATMANINFER_BI_ASSEMBLY_UTILS_HPP
