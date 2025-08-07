//
// Created by Mason on 2025/7/18.
// This is for GPT-2 Block
//

#pragma once

#include <runtime/neon/functions/BINEIntentAttention.hpp>
#include <runtime/neon/functions/BINEIntentFFNLayer.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include "bi_NEArithmeticAddition.h"
#include "bi_ne_copy.hpp"

namespace BatmanInfer {
    // Forward declaration
    class BIITensor;

    class BINEIntentGPTBlock final : public BIIFunction {
    public:
        explicit BINEIntentGPTBlock(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEIntentGPTBlock() : BINEIntentGPTBlock(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEIntentGPTBlock(const BINEIntentGPTBlock &) = delete;

        BINEIntentGPTBlock(BINEIntentGPTBlock &&) = delete;

        BINEIntentGPTBlock &operator=(const BINEIntentGPTBlock &) = delete;

        BINEIntentGPTBlock &operator=(BINEIntentGPTBlock &&) = delete;

        ~BINEIntentGPTBlock() override;

        void configure(BIITensor *input,
                       const BIITensor *ln_1_weight,
                       const BIITensor *ln_1_bias,
                       const BIITensor *c_attn_weights,
                       const BIITensor *c_attn_bias,
                       const BIITensor *o_attn_weights,
                       const BIITensor *o_attn_bias,
                       const BIITensor *fc_weights,
                       const BIITensor *fc_bias,
                       const BIITensor *proj_weights,
                       const BIITensor *proj_bias,
                       const BIITensor *ln_2_weight,
                       const BIITensor *ln_2_bias,
                       const BIActivationLayerInfo &act_info,
                       const PermutationVector &q_perm,
                       const PermutationVector &k_perm,
                       const PermutationVector &qkv_perm,
                       const size_t &hidden_size,
                       const size_t &max_seq_len,
                       const size_t &max_batch_size,
                       const size_t &seq_len,
                       const size_t &batch_size,
                       const int layer_idx,
                       BIITensor *output);

        /**
         * @brief 动态GPTBlock的配置
         * @param input 输入的张量信息
         * @param seq_len
         * @param batch_size
         * @param kv_caches_vec 输入的KV Cache的大小
         */
        void dynamic_configure(const BIITensor *input,
                               const size_t &seq_len,
                               const size_t &batch_size);

        void run() override;

        void prepare() override;

    private:
        BIMemoryGroup _memory_group; // 内存组管理

        // GPT-2 Block的算子
        BINEIntentAttention _attn_layer; // Attn模块算子
        BINEArithmeticAddition _add_layer;
        BINEIntentFFNLayer _mlp_layer; // 全连接层
        BINEArithmeticAddition _add_2_layer;
        BINECopy _copy_layer;

        // 最大张量管理器
        BITensor _attn_output;
        BITensor _attn_add_output;
        BITensor _mlp_output;
        BITensor _block_output;

        // 临时张量处理
        BITensor _sub_attn_output;
        BITensor _sub_add_output;
        BITensor _sub_mlp_output;
        BITensor _sub_block_output;

        // 临时张量信息
        BITensorInfo _sub_attn_output_info;
        BITensorInfo _sub_add_output_info;
        BITensorInfo _sub_mlp_output_info;
        BITensorInfo _sub_block_output_info;

    private:
        size_t _hidden_size{}; // 隐藏层大小
        size_t _max_seq_len{}; // 最大长度输入
        size_t _max_batch_size{}; // 一块的大小
        size_t _batch_size = 1;
        size_t _seq_len = 1;
        bool _is_prepared; // 是否已经完全初始化(预先把内存加载完)
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;
        int _layer_idx;
        bool _is_first_kv_cache = true; // 第一次使用KV Cache
        bool _is_first_gpt_block = false;
    };
}
