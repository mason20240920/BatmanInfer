//
// Created by Mason on 2025/1/23.
//

#pragma once

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/neon/functions/bi_ne_mat_mul.hpp>

#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include "BINELayerNormLayer.hpp"
#include "bi_NEArithmeticAddition.h"
#include "bi_NESoftmaxLayer.h"
#include "bi_ne_copy.hpp"
#include "bi_ne_permute.h"
#include "bi_ne_split.hpp"
#include "ne_pixel_wise_multiplication.hpp"

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BINEIntentAttention : public BIIFunction {
    public:
        explicit BINEIntentAttention(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEIntentAttention() : BINEIntentAttention(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEIntentAttention(const BINEIntentAttention &) = delete;

        BINEIntentAttention(BINEIntentAttention &&) = delete;

        BINEIntentAttention &operator=(const BINEIntentAttention &) = delete;

        BINEIntentAttention &operator=(BINEIntentAttention &&) = delete;

        ~BINEIntentAttention() override;

        void dynamic_configure(const BIITensor *input,
                               const size_t &seq_len,
                               const size_t &batch_size);

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
        * @param gamma_weights 权重张量，形状为 [input_size, num_units]. Data types supported: Same as @p input
        * @param c_attn_weights 偏置向量，形状为 [num_units]
        * @param c_attn_bias
        * @param o_attn_weights
        * @param o_attn_bias
        * @param q_perm
        * @param k_perm
        * @param qkv_perm
        * @param hidden_size
        * @param max_seq_len
        * @param max_batch_size
        * @param output
        */
        void configure(BIITensor *input,
                       const BIITensor *gamma_weights,
                       const BIITensor *ln_bias_weights,
                       const BIITensor *c_attn_weights,
                       const BIITensor *c_attn_bias,
                       const BIITensor *o_attn_weights,
                       const BIITensor *o_attn_bias,
                       const PermutationVector &q_perm,
                       const PermutationVector &k_perm,
                       const PermutationVector &qkv_perm,
                       const size_t &hidden_size,
                       const size_t &max_seq_len,
                       const size_t &max_batch_size,
                       const size_t &current_batch_size,
                       const size_t &current_seq_size,
                       BIITensor *output);

        /**
         * 验证函数
         * @param input
         * @param weights
         * @param bias
         * @param output
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
        BIMemoryGroup _memory_group; // 内存组管理

        // Attention模块算子
        BINELayerNormLayer _layer_norm_layer; // 归一化层
        BINEGEMM _c_attn_layer; // 进行channel-wise计算
        BINESplit _split_layer; // 切分层
        BINEReshapeLayer _reshape_q_layer, _reshape_k_layer, _reshape_v_layer;
        BINEPermute _transpose_q_layer, _transpose_k_layer, _transpose_v_layer;
        BINEMatMul _qk_bmm_layer;
        BINEArithmeticAddition _qk_add_layer;
        BINEPixelWiseMultiplication _divide_layer;
        BINESoftmaxLayer _softmax_layer;
        BINEMatMul _pv_bmm_layer;
        BINEPermute _pv_transpose_layer;
        BINEReshapeLayer _pv_reshape_layer;
        BINEGEMM _attn_o_gemm_layer;
        BINECopy _c_copy_layer;

        BITensor _sub_norm_output; // 归一化临时输出
        BITensor _sub_c_attn_output;
        BITensor _sub_query_states;
        BITensor _sub_key_states;
        BITensor _sub_value_states;
        BITensor _sub_reshape_q_states;
        BITensor _sub_reshape_k_states;
        BITensor _sub_reshape_v_states;
        // KV Cache合并的接口
        BITensor _sub_transpose_q_states;
        BITensor _sub_transpose_k_states;
        BITensor _sub_transpose_v_states;
        BITensor _sub_qk_bmm_output;
        BITensor _sub_add_output;
        BITensor _sub_add_weights;
        BITensor _sub_softmax_output;
        BITensor _sub_pv_bmm_output;
        BITensor _sub_pv_perm_output;
        BITensor _sub_pv_reshape_output;
        BITensor _sub_attn_o_output;
        BITensor _sub_divide_output;

        // 张量信息
        BITensorInfo _sub_norm_info;
        BITensorInfo _sub_c_attn_tensor_info;
        BITensorInfo _sub_qkv_states_info;
        BITensorInfo _sub_reshape_qkv_info;
        BITensorInfo _sub_transpose_q_info;
        BITensorInfo _sub_transpose_k_info;
        BITensorInfo _sub_transpose_v_info;
        BITensorInfo _sub_qk_bmm_output_info;
        BITensorInfo _sub_add_weights_info;
        BITensorInfo _sub_softmax_output_info;
        BITensorInfo _sub_pv_bmm_output_info;
        BITensorInfo _sub_pv_transpose_output_info;
        BITensorInfo _sub_pv_reshape_output_info;
        BITensorInfo _sub_attn_o_output_info;
        BITensorInfo _sub_add_output_info;

        // 中间内存管理的张量输出
        BITensor _norm_output; // 归一化输出值
        BITensor _c_attn_output; // QKV的第一次的矩阵计算
        BITensor _query_states;
        BITensor _key_states;
        BITensor _value_states;
        BITensor _reshape_q_states;
        BITensor _reshape_k_states;
        BITensor _reshape_v_states;
        BITensor _transpose_q_states;
        BITensor _transpose_k_states;
        BITensor _transpose_v_states;
        BITensor _qk_bmm_output;
        BITensor _add_output;
        BITensor _add_weights;
        BITensor _softmax_output;
        BITensor _pv_bmm_output;
        BITensor _pv_perm_output;
        BITensor _pv_reshape_output;
        BITensor _attn_o_output;
        BITensor _divide_output;
        BITensor _scale_tensor;

    private:
        size_t _hidden_size{}; // 隐藏层大小
        size_t _max_seq_len{}; // 最大长度输入
        size_t _max_batch_size{}; // 一块的大小
        size_t _batch_size = 1;
        size_t _seq_len = 1;
        bool _is_prepared; // 是否已经完全初始化(预先把内存加载完)
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;
    };
}
