//
// Created by Mason on 2025/7/18.
//
#include <runtime/neon/functions/BINEIntentGPTBlock.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <function_info/bi_MatMulInfo.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

namespace BatmanInfer {
    BINEIntentGPTBlock::~BINEIntentGPTBlock() = default;

    BINEIntentGPTBlock::BINEIntentGPTBlock(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)), _is_prepared(false), _layer_idx(0) {
    }

    void BINEIntentGPTBlock::configure(BIITensor *input,
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
                                        BIITensor *output) {
        _layer_idx = layer_idx;
        _is_first_gpt_block = layer_idx == 0;
        BI_COMPUTE_ERROR_ON_NULLPTR(input, ln_1_weight, c_attn_bias, c_attn_weights, output);
        BI_COMPUTE_LOG_PARAMS(input, ln_1_weight, c_attn_weights, output);

        _max_seq_len = max_seq_len; // 最大的值
        _hidden_size = hidden_size; // 隐藏层长度
        _max_batch_size = max_batch_size; // 最大块
        _seq_len = seq_len;
        _batch_size = batch_size;

        const auto common_shape = BITensorShape(_hidden_size, _max_seq_len, _max_batch_size);
        _attn_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));
        _attn_add_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));
        _mlp_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));
        _block_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));

        // 内存管理
        _memory_group.manage(&_attn_output);
        _memory_group.manage(&_attn_add_output);
        _memory_group.manage(&_mlp_output);
        _memory_group.manage(&_block_output);

        _attn_output.allocator()->allocate();
        _attn_add_output.allocator()->allocate();
        _mlp_output.allocator()->allocate();
        _block_output.allocator()->allocate();

        // 子张量管理
        const auto sub_common_shape = BITensorShape(_hidden_size, _seq_len, _batch_size);
        _sub_attn_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_attn_output_info.set_format(Format::F16);
        _sub_attn_output.allocator()->init(_sub_attn_output_info);

        _sub_add_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_add_output_info.set_format(Format::F16);
        _sub_add_output.allocator()->init(_sub_add_output_info);

        _sub_mlp_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_mlp_output_info.set_format(Format::F16);
        _sub_mlp_output.allocator()->init(_sub_mlp_output_info);

        _sub_block_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_block_output_info.set_format(Format::F16);
        _sub_block_output.allocator()->init(_sub_block_output_info);

        _attn_layer.configure(input,
                              ln_1_weight,
                              ln_1_bias,
                              c_attn_weights,
                              c_attn_bias,
                              o_attn_weights,
                              o_attn_bias,
                              q_perm,
                              k_perm,
                              qkv_perm,
                              hidden_size,
                              max_seq_len,
                              max_batch_size,
                              _batch_size,
                              _seq_len,
                              &_sub_attn_output);
        _add_layer.configure(input,
                             &_sub_attn_output,
                             &_sub_add_output,
                             BIConvertPolicy::SATURATE);
        _mlp_layer.configure(&_sub_add_output,
                             fc_weights,
                             fc_bias,
                             proj_weights,
                             proj_bias,
                             ln_2_weight,
                             ln_2_bias,
                             act_info,
                             &_sub_mlp_output,
                             max_batch_size,
                             max_seq_len,
                             _batch_size,
                             _seq_len);

        _add_2_layer.configure(&_sub_add_output, &_sub_mlp_output, output, BIConvertPolicy::SATURATE);
    }

    void BINEIntentGPTBlock::run() {
        prepare();

        _attn_layer.run();
        // 获取KV Cache Blocks
        // _attn_layer.get_kv_block_ids(kv_block_ids);
        _add_layer.run();
        _mlp_layer.run();
        _add_2_layer.run();
    }

    void BINEIntentGPTBlock::dynamic_configure(const BIITensor *input,
                                                const size_t &seq_len,
                                                const size_t &batch_size) {
        _batch_size = batch_size;
        _seq_len = seq_len;

        const auto sub_common_shape = BITensorShape(_hidden_size, seq_len, _batch_size);
        _sub_attn_output_info.set_tensor_shape(sub_common_shape);
        _sub_attn_output.allocator()->init(*_attn_output.allocator(), _sub_attn_output_info);

        _sub_add_output_info.set_tensor_shape(sub_common_shape);
        _sub_add_output.allocator()->init(*_attn_add_output.allocator(), _sub_add_output_info);

        _sub_mlp_output_info.set_tensor_shape(sub_common_shape);
        _sub_mlp_output.allocator()->init(*_mlp_output.allocator(), _sub_mlp_output_info);

        _sub_block_output_info.set_tensor_shape(sub_common_shape);
        _sub_block_output.allocator()->init(*_block_output.allocator(), _sub_block_output_info);

        _attn_layer.dynamic_configure(input, seq_len, batch_size);
        _add_layer.dynamic_configure(input, &_sub_attn_output, false);
        _mlp_layer.dynamic_configure(&_sub_add_output, batch_size, seq_len);
        _add_2_layer.dynamic_configure(&_sub_mlp_output, &_sub_add_output, false);
    }


    void BINEIntentGPTBlock::prepare() {
        if (!_is_prepared) {
            // 1. 先调用内存管理组(再进行sub tensor的内存分布, 申请开辟连续内存)
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            _sub_attn_output.allocator()->init(*_attn_output.allocator(), _sub_attn_output_info);
            _sub_add_output.allocator()->init(*_attn_add_output.allocator(), _sub_add_output_info);
            _sub_mlp_output.allocator()->init(*_mlp_output.allocator(), _sub_mlp_output_info);
            _sub_block_output.allocator()->init(*_block_output.allocator(), _sub_block_output_info);
            _is_prepared = true;
        }
    }
}