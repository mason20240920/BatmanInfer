//
// Created by Mason on 2025/2/9.
//

#include <runtime/neon/functions/BINEAttentionLowpLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <function_info/bi_MatMulInfo.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

#include "data/core/utils/quantization/asymm_helpers.hpp"
#include "kv_cache_manager/bi_kv_cache_manager.hpp"
#include "utils/utils.hpp"

namespace BatmanInfer {
    inline void invert_qinfo_offset(BITensor &t) {
        BIQuantizationInfo qinfo = t.info()->quantization_info();
        t.info()->set_quantization_info(BIQuantizationInfo(qinfo.scale()[0], -qinfo.offset()[0], qinfo.is_dynamic()));
    }

    BINEAttentionLowpLayer::~BINEAttentionLowpLayer() = default;

    BINEAttentionLowpLayer::BINEAttentionLowpLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)),
        _rms_norm_layer(),
        _quantization_layer(),
        _c_attn_layer(),
        _c_attn_o_stage(),
        _norm_output(),
        _q_norm_output(),
        _c_attn_s32_output(),
        _c_attn_q8_output(),
        _is_prepared(false) {
    }

    BIStatus
    BINEAttentionLowpLayer::validate(const BIITensorInfo *input,
                                     const BIITensorInfo *weights,
                                     const BIITensorInfo *bias,
                                     const BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F32, BIDataType::F16);

        BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 3);

        return BIStatus{};
    }

    void BINEAttentionLowpLayer::set_avail_lens(std::vector<size_t> *lens) {
        _avail_len = lens;
    }


    void BINEAttentionLowpLayer::dynamic_configure(const BIITensor *input,
                                                   const size_t &seq_len,
                                                   const size_t &batch_size,
                                                   std::vector<std::vector<unsigned int> > &kv_caches_vec) {
        _batch_size = batch_size;
        _seq_len = seq_len;
        _kv_decode_ids = std::move(kv_caches_vec);


        _sub_norm_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_norm_tensor.allocator()->init(*_norm_output.allocator(), _sub_norm_info);

        _sub_norm_q_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_norm_q_tensor.allocator()->init(*_q_norm_output.allocator(), _sub_norm_q_info);

        _sub_c_attn_s32_tensor_info.set_tensor_shape(BITensorShape(_hidden_size * 3, 1, batch_size));
        _sub_c_attn_s32_tensor.allocator()->init(*_c_attn_s32_output.allocator(), _sub_c_attn_s32_tensor_info);

        _sub_c_attn_q_info.set_tensor_shape(BITensorShape(_hidden_size * 3, 1, batch_size));
        _sub_c_attn_q8_tensor.allocator()->init(*_c_attn_q8_output.allocator(), _sub_c_attn_q_info);

        _sub_split_q_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_split_q_result_0.allocator()->init(*_split_q_result_0.allocator(), _sub_split_q_info);
        _sub_split_q_result_1.allocator()->init(*_split_q_result_1.allocator(), _sub_split_q_info);
        _sub_split_q_result_2.allocator()->init(*_split_q_result_2.allocator(), _sub_split_q_info);

        _sub_query_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_query_states.allocator()->init(*_query_states.allocator(), _sub_query_info);
        _sub_key_states.allocator()->init(*_key_states.allocator(), _sub_query_info);
        _sub_value_states.allocator()->init(*_value_states.allocator(), _sub_query_info);

        // _sub_q_query_info.set_tensor_shape(BITensorShape(_hidden_size, seq_len, batch_size));
        // _sub_q_query_states.allocator()->init(*_q_query_states.allocator(), _sub_q_query_info);
        //
        // _sub_q_key_info.set_tensor_shape(BITensorShape(_hidden_size, seq_len, batch_size));
        // _sub_q_key_states.allocator()->init(*_q_key_states.allocator(), _sub_q_key_info);

        _sub_q_value_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_q_value_states.allocator()->init(*_q_value_states.allocator(), _sub_q_value_info);

        _sub_reshape_q_info.set_tensor_shape(BITensorShape(64, 12, 1, batch_size));
        _sub_reshape_q_states.allocator()->init(*_reshape_q_states.allocator(), _sub_reshape_q_info);

        _sub_reshape_k_info.set_tensor_shape(BITensorShape(64, 12, 1, batch_size));
        _sub_reshape_k_states.allocator()->init(*_reshape_k_states.allocator(), _sub_reshape_k_info);

        _sub_reshape_v_info.set_tensor_shape(BITensorShape(64, 12, 1, batch_size));
        _sub_reshape_v_states.allocator()->init(*_reshape_v_states.allocator(), _sub_reshape_v_info);

        _sub_concat_reshape_k_info.set_tensor_shape(BITensorShape(64, 12, _seq_len, batch_size));
        _sub_concat_reshape_v_info.set_tensor_shape(BITensorShape(64, 12, _seq_len, batch_size));
        _sub_concat_reshape_k_states.allocator()->init(*_concat_reshape_k_states.allocator(),
                                                       _sub_concat_reshape_k_info);
        _sub_concat_reshape_v_states.allocator()->init(*_concat_reshape_v_states.allocator(),
                                                       _sub_concat_reshape_v_info);

        _sub_transpose_q_info.set_tensor_shape(BITensorShape(64, 1, 12, _batch_size));
        _sub_transpose_q_result.allocator()->init(*_transpose_q_result.allocator(), _sub_transpose_q_info);

        _sub_transpose_k_info.set_tensor_shape(BITensorShape(_seq_len, 64, 12, _batch_size));
        _sub_transpose_k_result.allocator()->init(*_transpose_k_result.allocator(), _sub_transpose_k_info);

        _sub_transpose_v_info.set_tensor_shape(BITensorShape(64, _seq_len, 12, _batch_size));
        _sub_transpose_v_result.allocator()->init(*_transpose_v_result.allocator(), _sub_transpose_v_info);

        _sub_qk_bmm_output_info.set_tensor_shape(BITensorShape(_seq_len, 1, 12, _batch_size));
        _sub_qk_bmm_output.allocator()->init(*_qk_bmm_output.allocator(), _sub_qk_bmm_output_info);

        _sub_softmax_output_info.set_tensor_shape(BITensorShape(_seq_len, 1, 12, _batch_size));
        _sub_softmax_output.allocator()->init(*_softmax_output.allocator(), _sub_softmax_output_info);

        _sub_softmax_q_result_info.set_tensor_shape(BITensorShape(_seq_len, 1, 12, _batch_size));
        _sub_softmax_q_result.allocator()->init(*_q_softmax_output.allocator(), _sub_softmax_q_result_info);

        _sub_pv_bmm_output_info.set_tensor_shape(BITensorShape(64, 1, 12, _batch_size));
        _sub_pv_bmm_output.allocator()->init(*_pv_bmm_output.allocator(), _sub_pv_bmm_output_info);

        _sub_pv_transpose_output_info.set_tensor_shape(BITensorShape(64, 12, 1, _batch_size));
        _sub_pv_transpose_output
                .allocator()->init(*_pv_perm_output.allocator(), _sub_pv_transpose_output_info);

        _sub_pv_reshape_output_info.set_tensor_shape(BITensorShape(768, 1, _batch_size));
        _sub_pv_reshape_output.allocator()->init(*_pv_reshape_output.allocator(), _sub_pv_reshape_output_info);

        _sub_pv_deq_output_info.set_tensor_shape(BITensorShape(768, 1, _batch_size));
        _sub_pv_deq_output.allocator()->init(*_pv_deq_output.allocator(), _sub_pv_deq_output_info);

        _sub_attn_o_output_info.set_tensor_shape(BITensorShape(768, 1, _batch_size));
        _sub_attn_o_output.allocator()->init(*_attn_o_output.allocator(), _sub_attn_o_output_info);

        std::vector<BIITensor *> outputs = {
            &_sub_split_q_result_0,
            &_sub_split_q_result_1,
            &_sub_split_q_result_2
        };

        _rms_norm_layer.dynamic_configure(input);
        _quantization_layer.dynamic_configure(&_sub_norm_tensor);
        _c_attn_layer.dynamic_configure(&_sub_norm_q_tensor, &_sub_c_attn_s32_tensor);
        _c_attn_o_stage.dynamic_configure(&_sub_c_attn_s32_tensor);
        _split_layer.dynamic_configure(&_sub_c_attn_q8_tensor, outputs);
        _deq_q_layer.dynamic_configure(&_sub_split_q_result_0);
        _deq_k_layer.dynamic_configure(&_sub_split_q_result_1);
        _deq_v_layer.dynamic_configure(&_sub_split_q_result_2);
        // _quant_q_layer.dynamic_configure(&_sub_query_states);
        // _quant_k_layer.dynamic_configure(&_sub_key_states);
        _quant_v_layer.dynamic_configure(&_sub_value_states);
        _reshape_q_layer.dynamic_configure();
        _reshape_k_layer.dynamic_configure();
        _reshape_v_layer.dynamic_configure();
        _transpose_q_layer.dynamic_configure(&_sub_reshape_q_states, &_sub_transpose_q_result);
        _transpose_k_layer.dynamic_configure(&_sub_concat_reshape_k_states, &_sub_transpose_k_result);
        _transpose_v_layer.dynamic_configure(&_sub_concat_reshape_v_states, &_sub_transpose_v_result);
        BIMatMulInfo matmul_info; // No transpose for lhs or rhs
        matmul_info.adj_lhs(false).adj_rhs(false);
        // Define CpuMatMulSettings
        BICpuMatMulSettings settings;
        // Enable fast math for optimization
        settings = settings.fast_math(true);
        _sub_transpose_q_result.info()->set_are_values_constant(false);
        _sub_transpose_k_result.info()->set_are_values_constant(false);
        _qk_bmm_layer.dynamic_configure(&_sub_transpose_q_result, &_sub_transpose_k_result, &_sub_qk_bmm_output);
        _softmax_layer.dynamic_configure();
        _q_softmax_layer.dynamic_configure(&_sub_softmax_output);
        _pv_bmm_layer.dynamic_configure(&_sub_softmax_q_result, &_sub_transpose_v_result, &_sub_pv_bmm_output);
        _pv_transpose_layer.dynamic_configure(&_sub_pv_bmm_output, &_sub_pv_transpose_output);
        _pv_reshape_layer.dynamic_configure();
        _pv_dequantization_layer.dynamic_configure(&_sub_pv_reshape_output);
        _attn_o_gemm_layer.dynamic_configure();
        _c_copy_layer.dynamic_configure();
        _sub_softmax_q_result.info()->set_are_values_constant(false);
        _sub_transpose_v_result.info()->set_are_values_constant(false);
        // _pv_bmm_layer.configure(&_sub_softmax_q_result, &_sub_transpose_v_result, &_sub_pv_bmm_output, matmul_info,
        //                         settings);
    }

    void BINEAttentionLowpLayer::get_kv_block_ids(std::vector<unsigned int> &kv_block_ids) {
        kv_block_ids = std::move(_block_ids);
    }


    void BINEAttentionLowpLayer::configure(const BIITensor *input,
                                           const BIITensor *gamma_weights,
                                           const BIITensor *c_attn_weights,
                                           const BIITensor *c_attn_bias,
                                           const BIITensor *o_attn_weights,
                                           const BIITensor *o_attn_bias,
                                           const std::string &eos_weights_path,
                                           const float &gemm_i_scale,
                                           const int &gemm_i_zp,
                                           const float &attn_gemm_o_scale,
                                           const int &attn_gemm_o_zp,
                                           const float &query_q_scale,
                                           const int &query_q_zp,
                                           const float &value_q_scale,
                                           const int &value_q_zp,
                                           const float &key_q_scale,
                                           const int &key_q_zp,
                                           const float &softmax_out_scale,
                                           const int &softmax_out_zp,
                                           const float &pv_bmm_out_scale,
                                           const int &pv_bmm_out_zp,
                                           const PermutationVector &q_perm,
                                           const PermutationVector &k_perm,
                                           const PermutationVector &qkv_perm,
                                           const size_t &hidden_size,
                                           const size_t &max_seq_len,
                                           const size_t &batch_size,
                                           BIITensor *output
    ) {
        // // 结果判断
        // BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, gamma, output); // 输入的参数是否为空
        // BI_COMPUTE_ERROR_THROW_ON(BINEAttentionLowpLayer::validate(input->info(), weights->info(),
        //     bias->info(), output->info())); // 验证输入, 权重，偏置和输出信息
        // BI_COMPUTE_LOG_PARAMS(input, weights, bias, output); // 获取log的参数

        // 配置私有参数
        _max_seq_len = max_seq_len; // 最大的值
        _hidden_size = hidden_size; // 隐藏层长度
        _max_batch_size = batch_size; // 最大块
        _is_prepared = false; // 初始化标志，标识尚未准备好

        // 配置中间张量输出
        auto normal_shape = BITensorShape(_hidden_size, 1, _max_batch_size); // 默认输入和输出
        auto c_attn_shape = BITensorShape(_hidden_size * 3, 1, _max_batch_size); // c_attn计算的输出
        auto reshape_q_shape = BITensorShape(64, 12, 1, _max_batch_size);
        const auto concat_reshape_kv_shape = BITensorShape(64, 12, _max_seq_len, _max_batch_size);
        auto transpose_q_shape = BITensorShape(64, 1, 12, _max_batch_size);
        auto transpose_v_shape = BITensorShape(64, _max_seq_len, 12, _max_batch_size);
        auto transpose_k_shape = BITensorShape(_max_seq_len, 64, 12, _max_batch_size);
        auto qk_bmm_output_shape = BITensorShape(_max_seq_len, 1, 12, _max_batch_size);


        const auto _norm_q_info = BIQuantizationInfo(gemm_i_scale, gemm_i_zp);
        _norm_output.allocator()->init(BITensorInfo(normal_shape, 1, input->info()->data_type()));
        _q_norm_output.allocator()->
                init(BITensorInfo(normal_shape, 1, BIDataType::QASYMM8_SIGNED, _norm_q_info));
        _c_attn_s32_output.allocator()->init(BITensorInfo(c_attn_shape, 1, BIDataType::S32));
        const auto _c_attn_q_info = BIQuantizationInfo(attn_gemm_o_scale, attn_gemm_o_zp);
        _c_attn_q8_output.allocator()->init(BITensorInfo(c_attn_shape,
                                                         1,
                                                         BIDataType::QASYMM8_SIGNED,
                                                         _c_attn_q_info));
        _split_q_result_0.allocator()->init(BITensorInfo(normal_shape,
                                                         1,
                                                         BIDataType::QASYMM8_SIGNED,
                                                         _c_attn_q_info));
        _split_q_result_1.allocator()->init(BITensorInfo(normal_shape,
                                                         1,
                                                         BIDataType::QASYMM8_SIGNED,
                                                         _c_attn_q_info));
        _split_q_result_2.allocator()->init(BITensorInfo(normal_shape,
                                                         1,
                                                         BIDataType::QASYMM8_SIGNED,
                                                         _c_attn_q_info));
        _query_states.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::F16));
        _key_states.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::F16));
        _value_states.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::F16));
        // const auto _q_query_info = BIQuantizationInfo(query_q_scale, query_q_zp);
        // _q_query_states.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::QASYMM8_SIGNED, _q_query_info));
        // const auto _q_key_info = BIQuantizationInfo(key_q_scale, key_q_zp);
        // _q_key_states.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::QASYMM8_SIGNED, _q_key_info));
        const auto _q_value_info = BIQuantizationInfo(value_q_scale, value_q_zp);
        _eos_q_smooth_tensor = utils::create_type_tensor(eos_weights_path, BITensorShape(64, 12, 16),
                                                         BIDataType::F16);
        _q_value_states.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info));
        _reshape_q_states.allocator()->
                init(BITensorInfo(reshape_q_shape, 1, BIDataType::F16));
        _reshape_k_states.allocator()->
                init(BITensorInfo(reshape_q_shape, 1, BIDataType::F16));
        _reshape_v_states.allocator()->
                init(BITensorInfo(reshape_q_shape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info));
        _concat_reshape_k_states.allocator()->init(BITensorInfo(concat_reshape_kv_shape, 1, BIDataType::F16));
        _concat_reshape_v_states.allocator()->init(
            BITensorInfo(concat_reshape_kv_shape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info));

        _transpose_q_result.allocator()->init(BITensorInfo(transpose_q_shape,
                                                           1,
                                                           BIDataType::F16));
        _transpose_k_result.allocator()->init(BITensorInfo(transpose_k_shape,
                                                           1,
                                                           BIDataType::F16));
        _transpose_v_result.allocator()->init(BITensorInfo(transpose_v_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                           _q_value_info));
        _qk_bmm_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        _softmax_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        const auto _q_softmax_info = BIQuantizationInfo(softmax_out_scale, softmax_out_zp);
        _q_softmax_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                         _q_softmax_info));

        const auto pv_bmm_output_info = BIQuantizationInfo(pv_bmm_out_scale, pv_bmm_out_zp);
        _pv_bmm_output.allocator()->init(BITensorInfo(transpose_q_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                      pv_bmm_output_info));

        _pv_perm_output.allocator()->init(BITensorInfo(reshape_q_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                       pv_bmm_output_info));
        _pv_reshape_output.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                          pv_bmm_output_info));
        _pv_deq_output.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::F16));
        _attn_o_output.allocator()->init(BITensorInfo(normal_shape, 1, BIDataType::F16));


        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_q_norm_output);
        _memory_group.manage(&_c_attn_s32_output);
        _memory_group.manage(&_c_attn_q8_output);
        _memory_group.manage(&_split_q_result_0);
        _memory_group.manage(&_split_q_result_1);
        _memory_group.manage(&_split_q_result_2);
        _memory_group.manage(&_query_states);
        _memory_group.manage(&_key_states);
        _memory_group.manage(&_value_states);
        // _memory_group.manage(&_q_query_states);
        // _memory_group.manage(&_q_key_states);
        _memory_group.manage(&_q_value_states);
        _memory_group.manage(&_reshape_q_states);
        _memory_group.manage(&_reshape_k_states);
        _memory_group.manage(&_reshape_v_states);
        _memory_group.manage(&_concat_reshape_k_states);
        _memory_group.manage(&_concat_reshape_v_states);
        _memory_group.manage(&_transpose_q_result);
        _memory_group.manage(&_transpose_k_result);
        _memory_group.manage(&_transpose_v_result);
        _memory_group.manage(&_qk_bmm_output);
        _memory_group.manage(&_softmax_output);
        _memory_group.manage(&_q_softmax_output);
        _memory_group.manage(&_pv_bmm_output);
        _memory_group.manage(&_pv_perm_output);
        _memory_group.manage(&_pv_reshape_output);
        _memory_group.manage(&_pv_deq_output);
        _memory_group.manage(&_attn_o_output);
        // _memory_group.manage(&_quantization_output);
        // _memory_group.manage(&_dequantization_output);

        _norm_output.allocator()->allocate();
        _q_norm_output.allocator()->allocate();
        _c_attn_s32_output.allocator()->allocate();
        _c_attn_q8_output.allocator()->allocate();
        _split_q_result_0.allocator()->allocate();
        _split_q_result_1.allocator()->allocate();
        _split_q_result_2.allocator()->allocate();
        _query_states.allocator()->allocate();
        _key_states.allocator()->allocate();
        _value_states.allocator()->allocate();
        // _q_query_states.allocator()->allocate();
        // _q_key_states.allocator()->allocate();
        _q_value_states.allocator()->allocate();
        _reshape_q_states.allocator()->allocate();
        _reshape_k_states.allocator()->allocate();
        _reshape_v_states.allocator()->allocate();
        _concat_reshape_k_states.allocator()->allocate();
        _concat_reshape_v_states.allocator()->allocate();
        _transpose_q_result.allocator()->allocate();
        _transpose_k_result.allocator()->allocate();
        _transpose_v_result.allocator()->allocate();
        _qk_bmm_output.allocator()->allocate();
        _softmax_output.allocator()->allocate();
        _q_softmax_output.allocator()->allocate();
        _pv_bmm_output.allocator()->allocate();
        _pv_perm_output.allocator()->allocate();
        _pv_reshape_output.allocator()->allocate();
        _pv_deq_output.allocator()->allocate();
        _attn_o_output.allocator()->allocate();
        // _quantization_output.allocator()->allocate();
        // _dequantization_output.allocator()->allocate();

        // 首次初始化
        const auto _sub_norm_shape = BITensorShape(_hidden_size, 1, _batch_size);
        _sub_norm_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_norm_info.set_format(Format::F16);
        _sub_norm_tensor.allocator()->init(_sub_norm_info);

        _sub_norm_q_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::QASYMM8_SIGNED, _norm_q_info);
        _sub_norm_q_info.set_format(Format::S8);
        _sub_norm_q_tensor.allocator()->init(_sub_norm_q_info);

        const auto _sub_c_attn_shape = BITensorShape(_hidden_size * 3, 1, _batch_size);
        _sub_c_attn_s32_tensor_info = BITensorInfo(_sub_c_attn_shape, 1, BIDataType::S32);
        _sub_c_attn_s32_tensor_info.set_format(Format::S32);
        _sub_c_attn_s32_tensor.allocator()->init(_sub_c_attn_s32_tensor_info);

        _sub_c_attn_q_info = BITensorInfo(_sub_c_attn_shape, 1, BIDataType::QASYMM8_SIGNED, _c_attn_q_info);
        _sub_c_attn_q_info.set_format(Format::S8);
        _sub_c_attn_q8_tensor.allocator()->init(_sub_c_attn_q_info);

        _sub_split_q_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::QASYMM8_SIGNED, _c_attn_q_info);
        _sub_split_q_info.set_format(Format::S8);
        _sub_split_q_result_0.allocator()->init(_sub_split_q_info);
        _sub_split_q_result_1.allocator()->init(_sub_split_q_info);
        _sub_split_q_result_2.allocator()->init(_sub_split_q_info);

        _sub_query_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_query_info.set_format(Format::F16);
        _sub_query_states.allocator()->init(_sub_query_info);
        _sub_key_states.allocator()->init(_sub_query_info);
        _sub_value_states.allocator()->init(_sub_query_info);


        // _sub_q_query_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::QASYMM8_SIGNED, _q_query_info);
        // _sub_q_query_info.set_format(Format::S8);
        // _sub_q_query_states.allocator()->init(_sub_q_query_info);

        // _sub_q_key_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::QASYMM8_SIGNED, _q_key_info);
        // _sub_q_key_info.set_format(Format::S8);
        // _sub_q_key_states.allocator()->init(_sub_q_key_info);

        _sub_q_value_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info);
        _sub_q_value_info.set_format(Format::S8);
        _sub_q_value_states.allocator()->init(_sub_q_value_info);

        const auto sub_reshape_q_shape = BITensorShape(64, 12, 1, _batch_size);
        _sub_reshape_q_info = BITensorInfo(sub_reshape_q_shape, 1, BIDataType::F16);
        _sub_reshape_q_info.set_format(Format::F16);
        _sub_reshape_q_states.allocator()->init(_sub_reshape_q_info);

        const auto sub_concat_qkv_reshape = BITensorShape(64, 12, _seq_len, _batch_size);
        _sub_concat_reshape_k_info = BITensorInfo(sub_concat_qkv_reshape, 1, BIDataType::F16);
        _sub_concat_reshape_k_info.set_format(Format::F16);;
        _sub_concat_reshape_k_states.allocator()->init(_sub_concat_reshape_k_info);

        _sub_concat_reshape_v_info = BITensorInfo(sub_concat_qkv_reshape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info);
        _sub_concat_reshape_v_info.set_format(Format::S8);;
        _sub_concat_reshape_v_states.allocator()->init(_sub_concat_reshape_v_info);

        _sub_reshape_k_info = BITensorInfo(sub_reshape_q_shape, 1, BIDataType::F16);
        _sub_reshape_k_info.set_format(Format::F16);
        _sub_reshape_k_states.allocator()->init(_sub_reshape_k_info);

        _sub_reshape_v_info = BITensorInfo(sub_reshape_q_shape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info);
        _sub_reshape_v_info.set_format(Format::S8);
        _sub_reshape_v_states.allocator()->init(_sub_reshape_v_info);

        const auto sub_transpose_q_shape = BITensorShape(64, 1, 12, _batch_size);
        _sub_transpose_q_info = BITensorInfo(sub_transpose_q_shape, 1, BIDataType::F16);
        _sub_transpose_q_info.set_format(Format::F16);
        _sub_transpose_q_result.allocator()->init(_sub_transpose_q_info);

        const auto sub_transpose_k_shape = BITensorShape(_seq_len, 64, 12, _batch_size);
        _sub_transpose_k_info = BITensorInfo(sub_transpose_k_shape, 1, BIDataType::F16);
        _sub_transpose_k_info.set_format(Format::F16);
        _sub_transpose_k_result.allocator()->init(_sub_transpose_k_info);

        const auto sub_transpose_v_shape = BITensorShape(64, _seq_len, 12, _batch_size);
        _sub_transpose_v_info = BITensorInfo(sub_transpose_v_shape, 1, BIDataType::QASYMM8_SIGNED, _q_value_info);
        _sub_transpose_v_info.set_format(Format::S8);
        _sub_transpose_v_result.allocator()->init(_sub_transpose_v_info);

        const auto sub_qk_bmm_output_shape = BITensorShape(_seq_len, 1, 12, _batch_size);
        _sub_qk_bmm_output_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::F16);
        _sub_qk_bmm_output_info.set_format(Format::F16);
        _sub_qk_bmm_output.allocator()->init(_sub_qk_bmm_output_info);

        _sub_softmax_output_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::F16);
        _sub_softmax_output_info.set_format(Format::F16);
        _sub_softmax_output.allocator()->init(_sub_softmax_output_info);

        _sub_softmax_q_result_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                  _q_softmax_info);
        _sub_softmax_q_result_info.set_format(Format::S8);
        _sub_softmax_q_result.allocator()->init(_sub_softmax_q_result_info);

        _sub_pv_bmm_output_info = BITensorInfo(sub_transpose_q_shape, 1, BIDataType::QASYMM8_SIGNED,
                                               pv_bmm_output_info);
        _sub_pv_bmm_output_info.set_format(Format::S8);
        _sub_pv_bmm_output.allocator()->init(_sub_pv_bmm_output_info);

        _sub_pv_transpose_output_info = BITensorInfo(sub_reshape_q_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                     pv_bmm_output_info);
        _sub_pv_transpose_output_info.set_format(Format::S8);
        _sub_pv_transpose_output.allocator()->init(_sub_pv_transpose_output_info);

        _sub_pv_reshape_output_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::QASYMM8_SIGNED, pv_bmm_output_info);
        _sub_pv_reshape_output_info.set_format(Format::S8);
        _sub_pv_reshape_output.allocator()->init(_sub_pv_reshape_output_info);

        _sub_pv_deq_output_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_pv_deq_output_info.set_format(Format::F16);
        _sub_pv_deq_output.allocator()->init(_sub_pv_deq_output_info);

        _sub_attn_o_output_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_attn_o_output_info.set_format(Format::F16);
        _sub_attn_o_output.allocator()->init(_sub_attn_o_output_info);

        // 配置量化信息
        _rms_norm_layer.configure(input, gamma_weights, &_sub_norm_tensor);
        _quantization_layer.configure(&_sub_norm_tensor, &_sub_norm_q_tensor);
        invert_qinfo_offset(_sub_norm_q_tensor);
        const auto gemm_info = GEMMInfo(false,
                                        false,
                                        true,
                                        false,
                                        false,
                                        false,
                                        BIGEMMLowpOutputStageInfo(),
                                        false, false, false,
                                        BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
        _c_attn_layer.configure(&_sub_norm_q_tensor, c_attn_weights, nullptr, &_sub_c_attn_s32_tensor, gemm_info);
        BIGEMMLowpOutputStageInfo attn_qkv_o_stage;
        attn_qkv_o_stage.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        attn_qkv_o_stage.output_data_type = BIDataType::QASYMM8_SIGNED; // 设置输c出数据类型
        attn_qkv_o_stage.is_quantized_per_channel = true; // 因为权重是per-channel量化// 设置输出范围
        attn_qkv_o_stage.gemmlowp_offset = static_cast<int32_t>(attn_gemm_o_zp);
        attn_qkv_o_stage.gemmlowp_min_bound = -128; // 通常是-128
        attn_qkv_o_stage.gemmlowp_max_bound = 127; // 通常是127// 假设已有输入tensor的量化参数
        quantization::calculate_quantized_multipliers(_sub_norm_q_tensor.info()->quantization_info(),
                                                      c_attn_weights->info()->quantization_info(),
                                                      _sub_c_attn_q8_tensor.info()->quantization_info(),
                                                      attn_qkv_o_stage);
        _c_attn_o_stage.configure(&_sub_c_attn_s32_tensor, c_attn_bias, &_sub_c_attn_q8_tensor, attn_qkv_o_stage);
        std::vector<BIITensor *> outputs = {
            &_sub_split_q_result_0,
            &_sub_split_q_result_1,
            &_sub_split_q_result_2
        };
        _split_layer.configure(&_sub_c_attn_q8_tensor, outputs, 0);
        _deq_q_layer.configure(&_sub_split_q_result_0, &_sub_query_states);
        _deq_k_layer.configure(&_sub_split_q_result_1, &_sub_key_states);
        _deq_v_layer.configure(&_sub_split_q_result_2, &_sub_value_states);
        // _quant_q_layer.configure(&_sub_query_states, &_sub_q_query_states);
        // _quant_k_layer.configure(&_sub_key_states, &_sub_q_key_states);
        _quant_v_layer.configure(&_sub_value_states, &_sub_q_value_states);
        _reshape_q_layer.configure(&_sub_query_states, &_sub_reshape_q_states);
        _reshape_k_layer.configure(&_sub_key_states, &_sub_reshape_k_states);
        _reshape_v_layer.configure(&_sub_q_value_states, &_sub_reshape_v_states);
        _transpose_q_layer.configure(&_sub_reshape_q_states, &_sub_transpose_q_result, q_perm);
        _transpose_k_layer.configure(&_sub_concat_reshape_k_states, &_sub_transpose_k_result, k_perm);
        _transpose_v_layer.configure(&_sub_concat_reshape_v_states, &_sub_transpose_v_result, q_perm);
        _sub_transpose_q_result.info()->set_are_values_constant(false);
        _sub_transpose_k_result.info()->set_are_values_constant(false);
        BIMatMulInfo matmul_info; // No transpose for lhs or rhs
        matmul_info.adj_lhs(false).adj_rhs(false);
        // Define CpuMatMulSettings
        BICpuMatMulSettings settings;
        // Enable fast math for optimization
        settings = settings.fast_math(true);
        _qk_bmm_layer.configure(&_sub_transpose_q_result, &_sub_transpose_k_result, &_sub_qk_bmm_output, matmul_info,
                                settings);
        _softmax_layer.configure(&_sub_qk_bmm_output, &_sub_softmax_output);
        _q_softmax_layer.configure(&_sub_softmax_output, &_sub_softmax_q_result);
        _sub_softmax_q_result.info()->set_are_values_constant(false);
        _sub_transpose_v_result.info()->set_are_values_constant(false);
        _pv_bmm_layer.configure(&_sub_softmax_q_result, &_sub_transpose_v_result, &_sub_pv_bmm_output, matmul_info,
                                settings);
        _pv_transpose_layer.configure(&_sub_pv_bmm_output, &_sub_pv_transpose_output, qkv_perm);
        _pv_reshape_layer.configure(&_sub_pv_transpose_output, &_sub_pv_reshape_output);
        _pv_dequantization_layer.configure(&_sub_pv_reshape_output, &_sub_pv_deq_output);
        _attn_o_gemm_layer.configure(&_sub_pv_deq_output, o_attn_weights, o_attn_bias, &_sub_attn_o_output, 1.0f, 1.0f,
                                     gemm_info);
        _c_copy_layer.configure(&_sub_attn_o_output, output);
    }

    void BINEAttentionLowpLayer::run() {
        // BIIOFormatInfo format;
        // format.element_delim = ", "; // 元素之间用逗号分隔
        // format.row_delim = "\n"; // 每行换行
        // format.align_columns = true; // 对齐列
        prepare(); // 内存分配管理

        // 运行计算
        _rms_norm_layer.run(); // 归一化计算
        _quantization_layer.run();
        // // invert_qinfo_offset(_sub_norm_q_tensor);
        _c_attn_layer.run();
        _c_attn_o_stage.run();
        _split_layer.run();
        _deq_q_layer.run();
        _deq_k_layer.run();
        _deq_v_layer.run();

        // _quant_q_layer.run();
        // _quant_k_layer.run();
        _quant_v_layer.run();
        _reshape_q_layer.run();
        _reshape_k_layer.run();
        _reshape_v_layer.run();
        store_kv_cache();
        concat_kv_cache();
        restruct_q_tensor();
        _transpose_q_layer.run();
        _transpose_k_layer.run();
        _transpose_v_layer.run();
        _qk_bmm_layer.run();
        _softmax_layer.run();
        _q_softmax_layer.run();
        _pv_bmm_layer.run();
        _pv_transpose_layer.run();
        _pv_reshape_layer.run();
        _pv_dequantization_layer.run();
        _attn_o_gemm_layer.run();
        _c_copy_layer.run();
    }

    void BINEAttentionLowpLayer::prepare() {
        if (!_is_prepared) {
            // 1. 先调用内存管理组(再进行子向量的内存分布，因为之前没有申请开辟连续内存)
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            _sub_norm_tensor.allocator()->init(*_norm_output.allocator(), _sub_norm_info);
            _sub_norm_q_tensor.allocator()->init(*_q_norm_output.allocator(), _sub_norm_q_info);
            _sub_c_attn_s32_tensor.allocator()->init(*_c_attn_s32_output.allocator(), _sub_c_attn_s32_tensor_info);
            _sub_c_attn_q8_tensor.allocator()->init(*_c_attn_q8_output.allocator(), _sub_c_attn_q_info);
            _sub_split_q_result_0.allocator()->init(*_split_q_result_0.allocator(), _sub_split_q_info);
            _sub_split_q_result_1.allocator()->init(*_split_q_result_1.allocator(), _sub_split_q_info);
            _sub_split_q_result_2.allocator()->init(*_split_q_result_2.allocator(), _sub_split_q_info);
            _sub_query_states.allocator()->init(*_query_states.allocator(), _sub_query_info);
            _sub_key_states.allocator()->init(*_key_states.allocator(), _sub_query_info);
            _sub_value_states.allocator()->init(*_value_states.allocator(), _sub_query_info);
            // _sub_q_query_states.allocator()->init(*_q_query_states.allocator(), _sub_q_query_info);
            _sub_q_value_states.allocator()->init(*_q_value_states.allocator(), _sub_q_value_info);
            // _sub_q_key_states.allocator()->init(*_q_key_states.allocator(), _sub_q_key_info);
            _sub_reshape_q_states.allocator()->init(*_reshape_q_states.allocator(), _sub_reshape_q_info);
            _sub_reshape_v_states.allocator()->init(*_reshape_v_states.allocator(), _sub_reshape_v_info);
            _sub_reshape_k_states.allocator()->init(*_reshape_k_states.allocator(), _sub_reshape_k_info);
            _sub_concat_reshape_k_states.allocator()->
                    init(*_concat_reshape_k_states.allocator(), _sub_concat_reshape_k_info);
            _sub_concat_reshape_v_states.allocator()->
                    init(*_concat_reshape_v_states.allocator(), _sub_concat_reshape_v_info);
            _sub_transpose_q_result.allocator()->init(*_transpose_q_result.allocator(), _sub_transpose_q_info);
            _sub_transpose_k_result.allocator()->init(*_transpose_k_result.allocator(), _sub_transpose_k_info);
            _sub_transpose_v_result.allocator()->init(*_transpose_v_result.allocator(), _sub_transpose_v_info);
            _sub_qk_bmm_output.allocator()->init(*_qk_bmm_output.allocator(), _sub_qk_bmm_output_info);
            _sub_softmax_output.allocator()->init(*_softmax_output.allocator(), _sub_softmax_output_info);
            _sub_softmax_q_result.allocator()->init(*_q_softmax_output.allocator(), _sub_softmax_q_result_info);
            _sub_pv_bmm_output.allocator()->init(*_pv_bmm_output.allocator(), _sub_pv_bmm_output_info);
            _sub_pv_transpose_output.allocator()->init(*_pv_perm_output.allocator(), _sub_pv_transpose_output_info);
            _sub_pv_reshape_output.allocator()->init
                    (*_pv_reshape_output.allocator(), _sub_pv_reshape_output_info);
            _sub_pv_deq_output.allocator()->init(*_pv_deq_output.allocator(), _sub_pv_deq_output_info);
            _sub_attn_o_output.allocator()->init(*_attn_o_output.allocator(), _sub_attn_o_output_info);
            _is_prepared = true;
        }
    }

    void BINEAttentionLowpLayer::store_kv_cache() {
        _block_ids.clear(); // 如果首次的话只会存入<s>的首字符
        if (_is_first_kv_cache) {
            const auto root_id = KVCacheManager::getInstance().root_id();
            KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_k_states.buffer(), root_id,0, true, true);
            KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_v_states.buffer(), root_id,0, false, true);

            _is_first_kv_cache = false;
            _block_ids.emplace_back(root_id);
            return;
        }
        auto batch_index = 0;
        // 判断当前的batch_size, 先根据batch size分配一组block_id
        for (const auto &decode_list: _kv_decode_ids) {
            auto block_ids = KVCacheManager::getInstance().alloc_decode_next(
                decode_list[0], decode_list.size() - 1, decode_list);
            // 进行内存值拷贝
            for (const auto &block_id: block_ids) {
                KVCacheManager::getInstance().
                        memcpy_decode_buffer(_sub_reshape_k_states.buffer(), block_id, batch_index, true, true);
                KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_v_states.buffer(),
                                                                   block_id,
                                                                   batch_index,
                                                                   false,
                                                                   true);
                _block_ids.emplace_back(block_id);
            }
            batch_index++;
        }
    }

    void BINEAttentionLowpLayer::concat_kv_cache() {
        std::vector<PhysicalBlock *> blocks{};
        std::vector<PhysicalBlock *> eos_blocks{};
        for (const auto &decode_list: _kv_decode_ids) {
            const auto block_id = decode_list[0];
            std::vector<unsigned int> decode_ids{};
            KVCacheManager::getInstance().decode_sequence_lst(block_id, decode_ids); // 获取合并的Decodes
            KVCacheManager::getInstance().decode_sequence_blocks(decode_ids, blocks, _seq_len);
        }
        KVCacheManager::getInstance().decode_eos_lst(eos_blocks, _seq_len);
        BIITensorPack pack;
        pack.add_tensor(ACL_SRC_0, &_sub_reshape_k_states);
        pack.add_tensor(ACL_SRC_1, &_sub_reshape_v_states);
        pack.add_tensor(ACL_DST_0, &_sub_concat_reshape_k_states);
        pack.add_tensor(ACL_DST_1, &_sub_concat_reshape_v_states);
        BINEScheduler::get().schedule_kv_concat(pack, blocks, *_avail_len);
        BINEScheduler::get().schedule_kv_full_fill(pack, eos_blocks, *_avail_len);
    }

    void BINEAttentionLowpLayer::restruct_q_tensor() {
        BIITensorPack pack;
        pack.add_tensor(ACL_SRC_0, &_sub_reshape_q_states);
        pack.add_tensor(ACL_SRC_1, &_eos_q_smooth_tensor);
        BINEScheduler::get().schedule_change_q(pack, *_avail_len, _seq_len);
    }
}
