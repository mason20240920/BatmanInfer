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

        void dynamic_configure(const BIITensor *input,
                               const size_t &seq_len,
                               const size_t &batch_size,
                               std::vector<std::vector<unsigned int> > &kv_caches_vec);

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
        void configure(BIITensor *input,
                       const BIITensor *gamma_weights,
                       const BIITensor *c_attn_weights,
                       const BIITensor *c_attn_bias,
                       const BIITensor *o_attn_weights,
                       const BIITensor *o_attn_bias,
                       const PermutationVector &q_perm,
                       const PermutationVector &k_perm,
                       const PermutationVector &qkv_perm,
                       const size_t &hidden_size,
                       const size_t &max_seq_len,
                       const size_t &batch_size,
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

        void get_kv_block_ids(std::vector<unsigned int> &kv_block_ids);

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
        BINERMSNormLayer _rms_norm_layer; // 归一化层
        BINEGEMM _c_attn_layer; // 进行channel-wise计算
        BINESplit _split_layer; // 切分层
        BINEReshapeLayer _reshape_q_layer, _reshape_k_layer, _reshape_v_layer;
        BINEPermute _transpose_q_layer, _transpose_k_layer, _transpose_v_layer;
        BINEMatMul _qk_bmm_layer;
        BINEArithmeticAddition _qk_add_layer;
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
        BITensor _sub_concat_reshape_k_states;
        BITensor _sub_concat_reshape_v_states;
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

        // 张量信息
        BITensorInfo _sub_norm_info;
        BITensorInfo _sub_c_attn_tensor_info;
        BITensorInfo _sub_qkv_states_info;
        BITensorInfo _sub_reshape_qkv_info;
        BITensorInfo _sub_concat_reshape_kv_info;
        BITensorInfo _sub_transpose_q_info;
        BITensorInfo _sub_transpose_k_info;
        BITensorInfo _sub_transpose_v_info;
        BITensorInfo _sub_qk_bmm_output_info;
        BITensorInfo _sub_add_output_info;
        BITensorInfo _sub_add_weights_info;
        BITensorInfo _sub_softmax_output_info;
        BITensorInfo _sub_pv_bmm_output_info;
        BITensorInfo _sub_pv_transpose_output_info;
        BITensorInfo _sub_pv_reshape_output_info;
        BITensorInfo _sub_attn_o_output_info;

        // 中间内存管理的张量输出
        BITensor _norm_output; // 归一化输出值
        BITensor _c_attn_output; // QKV的第一次的矩阵计算
        BITensor _query_states;
        BITensor _key_states;
        BITensor _value_states;
        BITensor _reshape_q_states;
        BITensor _reshape_k_states;
        BITensor _reshape_v_states;
        BITensor _concat_reshape_k_states;
        BITensor _concat_reshape_v_states;
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

    private:
        size_t _hidden_size{}; // 隐藏层大小
        size_t _max_seq_len{}; // 最大长度输入
        size_t _max_batch_size{}; // 一块的大小
        size_t _batch_size = 1;
        size_t _seq_len = 1;
        bool _is_prepared; // 是否已经完全初始化(预先把内存加载完)
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;
        bool _is_first_kv_cache = true; // 是否第一次KV Cache
        std::vector<std::vector<unsigned int> > _kv_decode_ids; // 进行kv cache的传递
        std::vector<unsigned int> _block_ids{};

    private:
        /**
         * @brief 存储每次计算的KV Caches
         */
        void store_kv_cache();

        /**
         * @brief 合并KV Cache缓存
         */
        void concat_kv_cache();
    };
}
