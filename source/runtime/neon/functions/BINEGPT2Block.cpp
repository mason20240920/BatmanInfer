//
// Created by Mason on 2025/7/18.
//
#include <runtime/neon/functions/BINEGPT2Block.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <function_info/bi_MatMulInfo.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

#include "kv_cache_manager/bi_kv_cache_manager.hpp"
#include "model_interface/gpt2_model.h"

namespace BatmanInfer {
    BINEGPT2Block::~BINEGPT2Block() = default;

    BINEGPT2Block::BINEGPT2Block(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)), _is_prepared(false) {
    }

    void BINEGPT2Block::configure(BIITensor *input,
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
                                  BIITensor *output) {
        _layer_idx = layer_idx;
        BI_COMPUTE_ERROR_ON_NULLPTR(input, ln_1_weight, c_attn_bias, c_attn_weights, output);
        BI_COMPUTE_LOG_PARAMS(input, ln_1_weight, c_attn_weights, output);

        _max_seq_len = max_seq_len; // 最大的值
        _hidden_size = hidden_size; // 隐藏层长度
        _max_batch_size = max_batch_size; // 最大块

        const auto common_shape = BITensorShape(_hidden_size, 1, _max_batch_size);
        _attn_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));
        _attn_add_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));
        _mlp_output.allocator()->init(BITensorInfo(common_shape, 1, BIDataType::F16));

        // 内存管理
        _memory_group.manage(&_attn_output);
        _memory_group.manage(&_attn_add_output);
        _memory_group.manage(&_mlp_output);

        _attn_output.allocator()->allocate();
        _attn_add_output.allocator()->allocate();
        _mlp_output.allocator()->allocate();

        // 子张量管理
        const auto sub_common_shape = BITensorShape(_hidden_size, 1, _batch_size);
        _sub_attn_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_attn_output_info.set_format(Format::F16);
        _sub_attn_output.allocator()->init(_sub_attn_output_info);

        _sub_add_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_add_output_info.set_format(Format::F16);
        _sub_add_output.allocator()->init(_sub_add_output_info);

        _sub_mlp_output_info = BITensorInfo(sub_common_shape, 1, BIDataType::F16);
        _sub_mlp_output_info.set_format(Format::F16);
        _sub_mlp_output.allocator()->init(_sub_mlp_output_info);

#ifdef FIX_VER
        _attn_lowp_layer.configure(input,
                                   ln_1_weight,
                                   c_attn_weights,
                                   c_attn_bias,
                                   o_attn_weights,
                                   o_attn_bias,
                                   eos_weights,
                                   0.05f,      // gemm_i_scale
                                   0,          // gemm_i_zp
                                   0.03f,      // attn_gemm_o_scale
                                   0,          // attn_gemm_o_zp
                                   0.04f,      // query_q_scale
                                   0,          // query_q_zp
                                   0.04f,      // value_q_scale
                                   0,          // value_q_zp
                                   0.04f,      // key_q_scale
                                   0,          // key_q_zp
                                   0.007f,     // softmax_out_scale
                                   0,          // softmax_out_zp
                                   0.02f,      // pv_bmm_out_scale
                                   0,          // pv_bmm_out_zp
                                   q_perm,
                                   k_perm,
                                   qkv_perm,
                                   hidden_size,
                                   max_seq_len,
                                   max_batch_size,
                                   layer_idx,
                                   &_sub_attn_output);
#else
        _attn_layer.configure(input,
                              ln_1_weight,
                              c_attn_weights,
                              c_attn_bias,
                              o_attn_weights,
                              o_attn_bias,
                              eos_weights,
                              q_perm,
                              k_perm,
                              qkv_perm,
                              hidden_size,
                              max_seq_len,
                              max_batch_size,
                              layer_idx,
                              &_sub_attn_output);
#endif
        _add_layer.configure(input,
                             &_sub_attn_output,
                             &_sub_add_output,
                             BIConvertPolicy::SATURATE);
#ifdef FIX_VER
        _mlp_layer.configure(&_sub_add_output,
                             0.05f,      // fc1_input_scale
                             0,          // fc1_input_zero_point
                             fc_weights,
                             fc_bias,
                             nullptr,    // c_fc_weight_qinfo
                             0.03f,      // fc1_output_scale
                             0,          // fc1_output_zero_point
                             0.04f,      // gelu_output_scale
                             0,          // gelu_output_zero_point
                             proj_weights,
                             proj_bias,
                             ln_2_weight,
                             &_sub_mlp_output,
                             max_batch_size,
                             1);
#else
        _mlp_layer.configure(&_sub_add_output,
                             fc_weights,
                             fc_bias,
                             proj_weights,
                             proj_bias,
                             ln_2_weight,
                             act_info,
                             &_sub_mlp_output,
                             hidden_size,
                             max_batch_size,
                             1);
#endif

        _add_2_layer.configure(&_sub_add_output, &_sub_mlp_output, output, BIConvertPolicy::SATURATE);
    }

    void BINEGPT2Block::run(const int layer_idx, std::vector<unsigned int> &kv_block_ids) {
        prepare();

#ifdef FIX_VER
        _attn_lowp_layer.run(layer_idx, kv_block_ids);
#else
        _attn_layer.run(layer_idx, kv_block_ids);
#endif
        // print_tensor(_sub_attn_output, "_sub_attn_output");

        if (0 == layer_idx) {
            // 获取KV Cache Blocks
#ifdef FIX_VER
            _attn_lowp_layer.get_kv_block_ids(kv_block_ids);
#else
            _attn_layer.get_kv_block_ids(kv_block_ids);
#endif
        }
        _add_layer.run();
        // print_tensor(_sub_add_output, "_sub_add_output");

        _mlp_layer.run();
        // print_tensor(_sub_mlp_output, "_sub_mlp_output");

        _add_2_layer.run();
    }

    void BINEGPT2Block::get_kv_block_ids(std::vector<unsigned int> &kv_block_ids) {
        _attn_layer.get_kv_block_ids(kv_block_ids);
    }

    void BINEGPT2Block::set_avail_lens(std::vector<size_t> *avail_lens) {
#ifdef FIX_VER
        _attn_lowp_layer.set_avail_lens(avail_lens);
#else
        _attn_layer.set_avail_lens(avail_lens);
#endif
    }




    void BINEGPT2Block::dynamic_configure(const BIITensor *input,
                                          const size_t &seq_len,
                                          const size_t &batch_size,
                                          const std::vector<std::vector<unsigned int> > &kv_caches_vec) {
        _batch_size = batch_size;
        _seq_len = seq_len;

        const auto sub_common_shape = BITensorShape(_hidden_size, 1, _batch_size);
        _sub_attn_output_info.set_tensor_shape(sub_common_shape);
        _sub_attn_output.allocator()->init(*_attn_output.allocator(), _sub_attn_output_info);

        _sub_add_output_info.set_tensor_shape(sub_common_shape);
        _sub_add_output.allocator()->init(*_attn_add_output.allocator(), _sub_add_output_info);

        _sub_mlp_output_info.set_tensor_shape(sub_common_shape);
        _sub_mlp_output.allocator()->init(*_mlp_output.allocator(), _sub_mlp_output_info);

#ifdef FIX_VER
        _attn_lowp_layer.dynamic_configure(input, seq_len, batch_size, kv_caches_vec);
#else
        _attn_layer.dynamic_configure(input, seq_len, batch_size, kv_caches_vec);
#endif
        _add_layer.dynamic_configure(input, &_sub_attn_output, false);
        _mlp_layer.dynamic_configure(&_sub_add_output, batch_size);
        _add_2_layer.dynamic_configure(&_sub_mlp_output, &_sub_add_output, false);
    }


    void BINEGPT2Block::prepare() {
        if (!_is_prepared) {
            // 1. 先调用内存管理组(再进行sub tensor的内存分布, 申请开辟连续内存)
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            _sub_attn_output.allocator()->init(*_attn_output.allocator(), _sub_attn_output_info);
            _sub_add_output.allocator()->init(*_attn_add_output.allocator(), _sub_add_output_info);
            _sub_mlp_output.allocator()->init(*_mlp_output.allocator(), _sub_mlp_output_info);
            _is_prepared = true;
        }
    }

    void BINEGPT2Block::print_tensor(const BatmanInfer::BITensor &tensor, const std::string &name , const BatmanInfer::BIIOFormatInfo::PrintRegion region) {
        std::cout << name << std::endl;
        BatmanInfer::BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = true; // 对齐列
        format.print_region = region;

        tensor.print(std::cout, format);
    }
}