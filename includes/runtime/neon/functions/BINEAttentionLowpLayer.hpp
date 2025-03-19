//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/neon/functions/bi_ne_mat_mul.hpp>

#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>
#include "bi_ne_copy.hpp"

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BINEAttentionLowpLayer : public BIIFunction {
    public:
        explicit BINEAttentionLowpLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEAttentionLowpLayer() : BINEAttentionLowpLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEAttentionLowpLayer(const BINEAttentionLowpLayer &) = delete;

        BINEAttentionLowpLayer(BINEAttentionLowpLayer &&) = delete;

        BINEAttentionLowpLayer &operator=(const BINEAttentionLowpLayer &) = delete;

        BINEAttentionLowpLayer &operator=(BINEAttentionLowpLayer &&) = delete;

        ~BINEAttentionLowpLayer() override;

        /**
        * 初始化函数
        * Valid data layouts:
        * - NHWC
        *
        * Valid data type configurations:
        * |src0   |src1   |src2   |src3   |dst0   |dst1   |
        * |:------|:------|:------|:------|:------|:------|
        * |F16    |F16    |F16    |F16    |F16    |F16    |
        * |F32    |F32    |F32    |F32    |F32    |F32    |
        * @param input 输入张量，形状为 [input_size, batch_size]。. Data types supported: F16/F32
        * @param weights 权重张量，形状为 [input_size, num_units]. Data types supported: Same as @p input
        * @param bias 偏置向量，形状为 [num_units]
        * @param add_weights 相加权重
        * @param perm 进行切换的perm
        * @param output 输出张量，形状为 [num_units, batch_size]
        * @param info 激活层参数，用于定义激活函数（如 ReLU、Tanh 等）
        */
        void configure(const BatmanInfer::BIITensor *input,
                       const BatmanInfer::BIITensor *weights,
                       const BatmanInfer::BIITensor *bias,
                       const BIITensor *scalar,
                       const BIITensor *add_weights,
                       const BIITensor *weights_second,
                       const BIITensor *bias_second,
                       const BIITensor *gamma,
                       const PermutationVector &perm,
                       const PermutationVector &perm2,
                       const PermutationVector &final_perm,
                       const size_t &hidden_size,
                       const size_t &max_seq_len,
                       const size_t &batch_size,
                       BatmanInfer::BIITensor *output);

        /**
         * 验证函数
         * @param input
         * @param weights
         * @param recurrent_weights
         * @param bias
         * @param hidden_state
         * @param output
         * @param info
         * @return
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *weights,
                                 const BIITensorInfo *bias,
                                 const BIITensorInfo *output);

        void run() override;

        void prepare() override;

    private:
        // 内存管理
        BIMemoryGroup _memory_group;

        // Attention模块算子
        BINERMSNormLayer _rms_norm_layer; // 归一化层
        BINEQuantizationLayer _quantization_layer; // 量化层
        BINEDequantizationLayer _dequantization_layer; // 反量化层
        BINECopy _copy_f; // 张量复制层

        // 中间内存管理的张量输出
        BITensor _norm_output; //1.归一化输出
        BITensor _quantization_output; // 量化输出
        BITensor _dequantization_output; // 2. 最终输出

        // 是否已经完全初始化
        size_t _hidden_size; // 隐藏层大小
        size_t _max_seq_len; // 最大长度输入
        size_t _batch_size; // 一块的大小
        bool _is_prepared; // 是否已经完全初始化
    };
}
