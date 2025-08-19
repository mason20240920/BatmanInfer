//
// Created by Mason on 2025/7/24.
//
#include <runtime/neon/functions/BINEIntentMGPTBlock.hpp>

#include "kv_cache_manager/bi_kv_cache_manager.hpp"

namespace BatmanInfer {
    BINEIntentMGPTBlock::~BINEIntentMGPTBlock() = default;

    BINEIntentMGPTBlock::BINEIntentMGPTBlock(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)), _is_prepared(false) {
    }


    void BINEIntentMGPTBlock::configure(BIITensor *input,
                                      const std::vector<BIGPTLayerConfig> &layer_configs,
                                      const BIGPTGlobalConfig &global_config,
                                      BIITensor *output) {
        // 1. 处理层数(查看多少层)
        _layer_num = layer_configs.size();
        _hidden_size = global_config.hidden_size;
        _max_batch_size = global_config.max_batch_size;
        _max_seq_len = global_config.max_seq_len;
        // 2. 创建中间张量(创建中间张量)
        _intermediate_tensors.reserve(_layer_num - 1);
        // 张量信息
        auto intermediate_tensor_info = BITensorInfo(
            BITensorShape(global_config.hidden_size, 1, global_config.max_batch_size),
            1,
            BIDataType::F16);
        _sub_intermediate_tensor_info = BITensorInfo(BITensorShape(global_config.hidden_size,
                                                                   1,
                                                                   _batch_size),
                                                     1,
                                                     BIDataType::F16);
        _sub_intermediate_tensor_info.set_format(Format::F16);
        // 如果层数只有一层
        if (_layer_num == 1) {
            auto gpt_block = std::make_unique<BINEIntentGPT2Block>();
            gpt_block->configure(input,
                                 layer_configs[0].ln_1_weight,
                                 layer_configs[0].ln_1_bias,
                                 layer_configs[0].c_attn_weights,
                                 layer_configs[0].c_attn_bias,
                                 layer_configs[0].o_attn_weights,
                                 layer_configs[0].o_attn_bias,
                                 layer_configs[0].fc_weights,
                                 layer_configs[0].fc_bias,
                                 layer_configs[0].proj_weights,
                                 layer_configs[0].proj_bias,
                                 layer_configs[0].ln_2_weight,
                                 layer_configs[0].ln_2_bias,
                                 layer_configs[0].act_info,
                                 global_config.q_perm,
                                 global_config.k_perm,
                                 global_config.qkv_perm,
                                 global_config.hidden_size,
                                 global_config.max_seq_len,
                                 global_config.max_batch_size,
                                 0,
                                 output);
            _layer_blocks.emplace_back(std::move(gpt_block));
        } else if (_layer_num > 1) {
            // 1. 先进行内存管理
            for (size_t i = 0; i < _layer_num - 1; i++) {
                // 先确定第一层的参数
                _intermediate_tensors.emplace_back();
                _intermediate_tensors[i].allocator()->init(intermediate_tensor_info);


                _sub_intermediate_tensors.emplace_back();
                _sub_intermediate_tensors[i].allocator()->init(_sub_intermediate_tensor_info);
            }

            for (auto &intermediate_t: _intermediate_tensors) {
                _memory_group.manage(&intermediate_t);
            }

            for (auto &intermediate_t: _intermediate_tensors) {
                intermediate_t.allocator()->allocate();
            }
            for (size_t i = 0; i < _layer_num; i++) {
                auto gpt_block = std::make_unique<BINEIntentGPT2Block>();
                if (i == 0) {
                    gpt_block->configure(input,
                                         layer_configs[i].ln_1_weight,
                                         layer_configs[i].ln_1_bias,
                                         layer_configs[i].c_attn_weights,
                                         layer_configs[i].c_attn_bias,
                                         layer_configs[i].o_attn_weights,
                                         layer_configs[i].o_attn_bias,
                                         layer_configs[i].fc_weights,
                                         layer_configs[i].fc_bias,
                                         layer_configs[i].proj_weights,
                                         layer_configs[i].proj_bias,
                                         layer_configs[i].ln_2_weight,
                                         layer_configs[i].ln_2_bias,
                                         layer_configs[i].act_info,
                                         global_config.q_perm,
                                         global_config.k_perm,
                                         global_config.qkv_perm,
                                         global_config.hidden_size,
                                         global_config.max_seq_len,
                                         global_config.max_batch_size,
                                         i,
                                         &_sub_intermediate_tensors.at(i));
                } else if (i == _layer_num - 1) {
                    // 最后一层的configure
                    gpt_block->configure(&_sub_intermediate_tensors.at(i - 1),
                                         layer_configs[i].ln_1_weight,
                                         layer_configs[i].ln_1_bias,
                                         layer_configs[i].c_attn_weights,
                                         layer_configs[i].c_attn_bias,
                                         layer_configs[i].o_attn_weights,
                                         layer_configs[i].o_attn_bias,
                                         layer_configs[i].fc_weights,
                                         layer_configs[i].fc_bias,
                                         layer_configs[i].proj_weights,
                                         layer_configs[i].proj_bias,
                                         layer_configs[i].ln_2_weight,
                                         layer_configs[i].ln_2_bias,
                                         layer_configs[i].act_info,
                                         global_config.q_perm,
                                         global_config.k_perm,
                                         global_config.qkv_perm,
                                         global_config.hidden_size,
                                         global_config.max_seq_len,
                                         global_config.max_batch_size,
                                         i,
                                         output);
                } else {
                    // 先确定第一层的参数
                    BITensor block_o_tensor;
                    _intermediate_tensors.push_back(std::move(block_o_tensor));
                    _intermediate_tensors[i].allocator()->init(intermediate_tensor_info);

                    BITensor sub_block_o_tensor;
                    sub_block_o_tensor.allocator()->init(_sub_intermediate_tensor_info);
                    _sub_intermediate_tensors.emplace_back(std::move(sub_block_o_tensor));
                    gpt_block->configure(&_sub_intermediate_tensors.at(i - 1),
                                         layer_configs[i].ln_1_weight,
                                         layer_configs[i].ln_1_bias,
                                         layer_configs[i].c_attn_weights,
                                         layer_configs[i].c_attn_bias,
                                         layer_configs[i].o_attn_weights,
                                         layer_configs[i].o_attn_bias,
                                         layer_configs[i].fc_weights,
                                         layer_configs[i].fc_bias,
                                         layer_configs[i].proj_weights,
                                         layer_configs[i].proj_bias,
                                         layer_configs[i].ln_2_weight,
                                         layer_configs[i].ln_2_bias,
                                         layer_configs[i].act_info,
                                         global_config.q_perm,
                                         global_config.k_perm,
                                         global_config.qkv_perm,
                                         global_config.hidden_size,
                                         global_config.max_seq_len,
                                         global_config.max_batch_size,
                                         i,
                                         &_sub_intermediate_tensors.at(i));
                }
                _layer_blocks.emplace_back(std::move(gpt_block));
            }
        }
    }

    template<size_t NumLayers>
    void BINEIntentMGPTBlock::configure_fixed(BIITensor *input,
                                            const std::array<BIGPTLayerConfig, NumLayers> &layer_configs,
                                            const BIGPTGlobalConfig &global_config,
                                            BIITensor *output) {
        std::vector<BIGPTLayerConfig> configs(layer_configs.begin(), layer_configs.end());
        configure(input, configs, global_config, output);
    }

    void BINEIntentMGPTBlock::dynamic_configure(const BIITensor *input,
                                              const size_t &seq_len,
                                              const size_t &batch_size) {
        _batch_size = batch_size;
        _sub_intermediate_tensor_info.set_tensor_shape(BITensorShape(_hidden_size,
                                                                     1,
                                                                     _batch_size));
        for (int i = 0; i < _sub_intermediate_tensors.size(); i++) {
            _sub_intermediate_tensors[i].allocator()->init(*_intermediate_tensors[i].allocator(),
                                                           _sub_intermediate_tensor_info);
        }

        for (int i = 0; i < _layer_blocks.size(); i++) {
            if (i == 0) {
                _layer_blocks[i]->dynamic_configure(input, seq_len, batch_size);
            } else {
                _layer_blocks[i]->dynamic_configure(&_sub_intermediate_tensors.at(i - 1),
                                                    seq_len,
                                                    batch_size);
            }
        }
    }


    void BINEIntentMGPTBlock::run() {
        prepare();
        for (const auto &layer: _layer_blocks) {
            layer->run();
        }
    }

    void BINEIntentMGPTBlock::prepare() {
        if (!_is_prepared) {
            // 1. 先调用内存管理组(再进行sub tensor的内存分布, 申请开辟连续内存)
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            // 进行内存分配
            for (int i = 0; i < _sub_intermediate_tensors.size(); i++) {
                _sub_intermediate_tensors[i].allocator()->init(*_intermediate_tensors[i].allocator(),
                                                               _sub_intermediate_tensor_info);
            }
            _is_prepared = true;
        }
    }
}