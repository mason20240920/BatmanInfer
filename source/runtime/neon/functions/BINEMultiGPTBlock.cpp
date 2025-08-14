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
                                      BIITensor *eos_weights,
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
                                 eos_weights,
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
                                         eos_weights,
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
                                         eos_weights,
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
                                         eos_weights,
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
                                            BIITensor *eos_weights,
                                            BIITensor *output) {
        std::vector<BIGPTLayerConfig> configs(layer_configs.begin(), layer_configs.end());
        configure(input, configs, global_config, eos_weights, output);
    }

    void BINEMultiGPTBlock::dynamic_configure(const BIITensor *input,
                                              const size_t &seq_len,
                                              const size_t &batch_size,
                                              const std::vector<std::vector<unsigned int> > &kv_caches_vec) {
        _batch_size = batch_size;
        _kv_decode_ids = std::move(kv_caches_vec);
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
        store_kv_cache();
        concat_kv_cache();
        for (const auto &layer: _layer_blocks) {
            layer->set_history_ids(&_kv_history_ids);
            layer->set_physical_blocks(&_physic_blocks);
            layer->run();
        }
    }

    void BINEMultiGPTBlock::get_kv_block_ids(std::vector<unsigned int> &kv_block_ids) {
        kv_block_ids = std::move(_block_ids);
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


    void BINEMultiGPTBlock::store_kv_cache() {
        _block_ids.clear();
        _kv_history_ids.clear();
        if (_is_first_kv_cache) {
            const auto root_id = KVCacheManager::getInstance().root_id();
            _is_first_kv_cache = false;
            _block_ids.emplace_back(root_id);
            _kv_history_ids.emplace_back(std::vector<unsigned int>{root_id});
            return;
        }
        // 判断当前的batch_size, 先根据batch size分配一组block_id
        for (const auto &decode_list: _kv_decode_ids) {
            auto block_ids = KVCacheManager::getInstance().alloc_decode_next(decode_list[0],
                                                                             decode_list.size() - 1,
                                                                             decode_list);
            _kv_history_ids.emplace_back(block_ids);
            // 进行内存值拷贝
            for (const auto &block_id: block_ids) {
                _block_ids.emplace_back(block_id);
            }
        }
    }

    void BINEMultiGPTBlock::concat_kv_cache() {
        _physic_blocks.clear();
        for (const auto &decode_list: _kv_decode_ids) {
            const auto block_id = decode_list[0];
            std::vector<unsigned int> decode_ids{};
            KVCacheManager::getInstance().decode_sequence_lst(block_id, decode_ids); // 获取合并的Decodes
            KVCacheManager::getInstance().decode_sequence_blocks(decode_ids, _physic_blocks, _seq_len);
        }
    }
}
