//
// Created by Mason on 2025/7/24.
//
#include <runtime/neon/functions/BINEMultiGPTBlock.hpp>

#include "kv_cache_manager/bi_kv_cache_manager.hpp"

namespace BatmanInfer {
    BINEMultiGPTBlock::~BINEMultiGPTBlock() = default;

    BINEMultiGPTBlock::BINEMultiGPTBlock(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)), _is_prepared(false) {
    }


    void BINEMultiGPTBlock::configure(BIITensor *input,
                                      const std::vector<BIGPTLayerConfig> &layer_configs,
                                      const BIGPTGlobalConfig &global_config,
                                      std::array<BITensor, 3> &eos_weights,
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
            auto gpt_block = std::make_unique<BINEGPT2Block>();
            gpt_block->configure(input,
                                 layer_configs[0].ln_1_weight,
                                 layer_configs[0].c_attn_weights,
                                 layer_configs[0].c_attn_bias,
                                 layer_configs[0].o_attn_weights,
                                 layer_configs[0].o_attn_bias,
                                 layer_configs[0].fc_weights,
                                 layer_configs[0].fc_bias,
                                 layer_configs[0].proj_weights,
                                 layer_configs[0].proj_bias,
                                 layer_configs[0].ln_2_weight,
                                 &eos_weights.at(0),
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
                auto gpt_block = std::make_unique<BINEGPT2Block>();
                if (i == 0) {
                    gpt_block->configure(input,
                                         layer_configs[i].ln_1_weight,
                                         layer_configs[i].c_attn_weights,
                                         layer_configs[i].c_attn_bias,
                                         layer_configs[i].o_attn_weights,
                                         layer_configs[i].o_attn_bias,
                                         layer_configs[i].fc_weights,
                                         layer_configs[i].fc_bias,
                                         layer_configs[i].proj_weights,
                                         layer_configs[i].proj_bias,
                                         layer_configs[i].ln_2_weight,
                                         &eos_weights.at(i),
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
                                         layer_configs[i].c_attn_weights,
                                         layer_configs[i].c_attn_bias,
                                         layer_configs[i].o_attn_weights,
                                         layer_configs[i].o_attn_bias,
                                         layer_configs[i].fc_weights,
                                         layer_configs[i].fc_bias,
                                         layer_configs[i].proj_weights,
                                         layer_configs[i].proj_bias,
                                         layer_configs[i].ln_2_weight,
                                         &eos_weights.at(i),
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
                    gpt_block->configure(&_sub_intermediate_tensors.at(i - 1),
                                         layer_configs[i].ln_1_weight,
                                         layer_configs[i].c_attn_weights,
                                         layer_configs[i].c_attn_bias,
                                         layer_configs[i].o_attn_weights,
                                         layer_configs[i].o_attn_bias,
                                         layer_configs[i].fc_weights,
                                         layer_configs[i].fc_bias,
                                         layer_configs[i].proj_weights,
                                         layer_configs[i].proj_bias,
                                         layer_configs[i].ln_2_weight,
                                         &eos_weights.at(i),
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
    void BINEMultiGPTBlock::configure_fixed(BIITensor *input,
                                            const std::array<BIGPTLayerConfig, NumLayers> &layer_configs,
                                            const BIGPTGlobalConfig &global_config,
                                            std::array<BITensor, NumLayers> &eos_weights,
                                            BIITensor *output) {
        std::vector<BIGPTLayerConfig> configs(layer_configs.begin(), layer_configs.end());
        configure(input, configs, global_config, eos_weights, output);
    }

    void BINEMultiGPTBlock::dynamic_configure(const BIITensor *input,
                                              const size_t &seq_len,
                                              const size_t &batch_size,
                                              const std::vector<std::vector<unsigned int> > &kv_caches_vec) {
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
                _layer_blocks[i]->dynamic_configure(input, seq_len, batch_size, kv_caches_vec);
            } else {
                _layer_blocks[i]->dynamic_configure(&_sub_intermediate_tensors.at(i - 1),
                                                    seq_len,
                                                    batch_size,
                                                    kv_caches_vec);
            }
        }
    }


    void BINEMultiGPTBlock::run() {
        prepare();
        // 有 N块 GPTBlock，对于第一块 GPTBlock，需要进行 KVCache 的选择，并将当前 Block 的 KV值写入到内存指定位置；
        // 对于后续 N-1块 GPTBlock，需要将已经选择好的 KVCache block_ids 进行传递，并将当前 Block 的 KV值写入到内存指定位置；
        std::vector<unsigned int> kv_block_ids;
        for (int i = 0; i < _layer_blocks.size(); ++i) {
            _layer_blocks.at(i)->run(i, kv_block_ids);
        }
    }

    void BINEMultiGPTBlock::get_kv_block_ids(std::vector<unsigned int> &kv_block_ids) {
        _layer_blocks.at(0)->get_kv_block_ids(kv_block_ids);
    }

    void BINEMultiGPTBlock::prepare() {
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

    void BINEMultiGPTBlock::set_avail_lens(std::vector<size_t> *avail_lens) const {
        for (const auto &layer: _layer_blocks) {
            layer->set_avail_lens(avail_lens);
        }
    }

}