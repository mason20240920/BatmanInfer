//
// Created by Mason on 2025/1/23.
//

#pragma once

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/neon/functions/bi_ne_mat_mul.hpp>

#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>
#include "bi_ne_copy.hpp"

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BINEAttentionLayer : public BIIFunction {
    public:
        explicit BINEAttentionLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEAttentionLayer() : BINEAttentionLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEAttentionLayer(const BINEAttentionLayer &) = delete;

        BINEAttentionLayer(BINEAttentionLayer &&) = delete;

        BINEAttentionLayer &operator=(const BINEAttentionLayer &) = delete;

        BINEAttentionLayer &operator=(BINEAttentionLayer &&) = delete;

        ~BINEAttentionLayer() override;

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
        void configure(const BIITensor *input,
                       const BIITensor *weights,
                       const BIITensor *bias,
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
                       BIITensor *output);

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

        /***
         * 设置输入的sequence长度
         * @param seq_len
         */
        void set_sequence_length(int seq_len);

        void run() override;

        void prepare() override;

    private:
        // 内存管理
        BIMemoryGroup _memory_group; // 内存组管理

        // Attention模块算子
        BINERMSNormLayer _normalization_layer; // 用于执行归一操作的层
        BINECopy _copy_f; // 张量复制层
        BINEGEMM _gemm_state_f; // 执行矩阵乘法(GEMM), 用于计算当前状态或隐藏层的现象变换
        BINESplit _split_layer; // 切分层
        BINEReshapeLayer _reshape_1_f; // 分支1的reshape
        BINEPermute _transpose_1_f; // 转置transpose
        BINEReshapeLayer _reshape_2_f; // 分支2的reshape
        BINEPermute _transpose_2_f; // 转置transpose
        BINEReshapeLayer _reshape_3_f; // 分支2的reshape
        BINEPermute _transpose_3_f; // 转置transpose
        BINEPixelWiseMultiplication _mul_1_f;
        BINEPixelWiseMultiplication _mul_2_f;
        BINEMatMul _matmul_1_f;
        BINEArithmeticAddition _add_f;
        BINESoftmaxLayerGeneric<false> _softmax_layer;
        BINEMatMul _matmul_2_f;
        BINEPermute _transpose_final_f; // 转置transpose
        BINEReshapeLayer _reshape_final_f; //
        BINEGEMM _gemm_final_f;

        // 中间内存管理的张量输出
        BITensor _norm_output; // 归一化输出值
        BITensor _gemm_output; // _gemm输出
        BITensor _split_result_0;
        BITensor _split_result_1;
        BITensor _split_result_2;
        BITensor _reshape_1_output;
        BITensor _transpose_1_output;
        BITensor _reshape_2_output;
        BITensor _transpose_2_output;
        BITensor _reshape_3_output;
        BITensor _transpose_3_output;
        BITensor _mul_1_output;
        BITensor _mul_2_output;
        BITensor _matmul_1_output;
        BITensor _add_output;
        BITensor _softmax_output;
        BITensor _matmul_2_output;
        BITensor _transpose_final_output;
        BITensor _reshape_final_output;
        BITensor _gemm_final_output;


        // 其他参数
        size_t _hidden_size; // 隐藏层大小
        size_t _max_seq_len; // 最大长度输入
        size_t _batch_size; // 一块的大小
        bool _is_prepared; // 是否已经完全初始化
    };
}
