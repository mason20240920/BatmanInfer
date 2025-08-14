//
// Created by Mason on 2025/7/18.
// This is for GPT-2 Block
//

#pragma once

#include <runtime/neon/functions/BINEAttentionLayer.hpp>
#include <runtime/neon/functions/BINEFeedForwardLayer.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
// #include <runtime/neon/bi_ne_functions.h>

#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include "bi_NEArithmeticAddition.h"
#include "bi_ne_copy.hpp"
#include "kv_cache_manager/block/physical_block.hpp"

namespace BatmanInfer {
    // Forward declaration
    class BIITensor;

    class BINEGPT2Block : public BIIFunction {
    public:
        explicit BINEGPT2Block(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEGPT2Block() : BINEGPT2Block(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEGPT2Block(const BINEGPT2Block &) = delete;

        BINEGPT2Block(BINEGPT2Block &&) = delete;

        BINEGPT2Block &operator=(const BINEGPT2Block &) = delete;

        BINEGPT2Block &operator=(BINEGPT2Block &&) = delete;

        ~BINEGPT2Block() override;

        void configure(BIITensor *input,
                       const BIITensor *ln_1_weight,
                       const BIITensor *c_attn_weights,
                       const BIITensor *c_attn_bias,
                       const BIITensor *o_attn_weights,
                       const BIITensor *o_attn_bias,
                       const BIITensor *fc_weights,
                       const BIITensor *fc_bias,
                       const BIITensor *proj_weights,
                       const BIITensor *proj_bias,
                       const BIITensor *ln_2_weight,
                       BIITensor *eos_weights,
                       const BIActivationLayerInfo &act_info,
                       const PermutationVector &q_perm,
                       const PermutationVector &k_perm,
                       const PermutationVector &qkv_perm,
                       const size_t &hidden_size,
                       const size_t &max_seq_len,
                       const size_t &max_batch_size,
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
                               const size_t &batch_size,
                               const std::vector<std::vector<unsigned int> > &kv_caches_vec);

        void run() override;

        void prepare() override;

        void set_history_ids(std::vector<std::vector<unsigned int> > *history_ids);

        void set_physical_blocks(std::vector<PhysicalBlock *> *physical_blocks);

        void set_avail_lens(std::vector<size_t> *avail_lens);

    private:
        BIMemoryGroup _memory_group; // 内存组管理

        // GPT-2 Block的算子
        BINEAttentionLayer _attn_layer; // Attn模块算子
        BINEArithmeticAddition _add_layer;
        BINEFeedForwardLayer _mlp_layer; // 全连接层
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
