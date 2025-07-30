//
// Created by Mason on 2025/1/23.
//

#include <runtime/neon/functions/BINEAttentionLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <function_info/bi_MatMulInfo.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

#include "kv_cache_manager/bi_kv_cache_manager.hpp"

namespace BatmanInfer {
    BINEAttentionLayer::~BINEAttentionLayer() = default;

    BINEAttentionLayer::BINEAttentionLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)),

        // _rms_norm_layer(),
        // _c_attn_layer(),
        // _norm_output(),
        // _gemm_state_f(),
        // _split_layer(),
        // _reshape_1_f(),
        // _transpose_1_f(),
        // _reshape_2_f(),
        // _transpose_2_f(),
        // _reshape_3_f(),
        // _transpose_3_f(),
        // _mul_1_f(),
        // _mul_2_f(),
        // _matmul_1_f(),
        // _matmul_2_f(),
        // _add_f(),
        // _softmax_layer(),
        // _transpose_final_f(),
        // _reshape_final_f(),
        // _gemm_final_f(),
        // _norm_output(),
        // _gemm_output(),
        // _split_result_0(),
        // _split_result_1(),
        // _split_result_2(),
        // _reshape_1_output(),
        // _transpose_1_output(),
        // _reshape_2_output(),
        // _transpose_2_output(),
        // _reshape_3_output(),
        // _transpose_3_output(),
        // _mul_1_output(),
        // _mul_2_output(),
        // _matmul_1_output(),
        // _add_output(),
        // _softmax_output(),
        // _matmul_2_output(),
        // _transpose_final_output(),
        // _reshape_final_output(),
        // _gemm_final_output(),
        _is_prepared(
            false
        ) {
    }

    BIStatus
    BINEAttentionLayer::validate(const BatmanInfer::BIITensorInfo *input,
                                 const BatmanInfer::BIITensorInfo *weights,
                                 const BatmanInfer::BIITensorInfo *bias,
                                 const BatmanInfer::BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F16, BIDataType::F32);

        BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() == 1);

        return BIStatus{};
    }

    void BINEAttentionLayer::set_avail_lens(std::vector<size_t> *lens) {
        _avail_len = lens;
    }


    void BINEAttentionLayer::dynamic_configure(const BIITensor *input,
                                               const size_t &seq_len,
                                               const size_t &batch_size,
                                               std::vector<std::vector<unsigned int> > &kv_caches_vec) {
        _batch_size = batch_size;
        _seq_len = seq_len;
        _kv_decode_ids = std::move(kv_caches_vec);

        _sub_norm_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_info);

        _sub_c_attn_tensor_info.set_tensor_shape(BITensorShape(_hidden_size * 3, 1, batch_size));
        _sub_c_attn_output.allocator()->init(*_c_attn_output.allocator(), _sub_c_attn_tensor_info);

        _sub_qkv_states_info.set_tensor_shape(BITensorShape(_hidden_size, 1, batch_size));
        _sub_query_states.allocator()->init(*_query_states.allocator(), _sub_qkv_states_info);
        _sub_key_states.allocator()->init(*_key_states.allocator(), _sub_qkv_states_info);
        _sub_value_states.allocator()->init(*_value_states.allocator(), _sub_qkv_states_info);

        _sub_reshape_qkv_info.set_tensor_shape(BITensorShape(64, 12, 1, batch_size));
        _sub_reshape_q_states.allocator()->init(*_reshape_q_states.allocator(), _sub_reshape_qkv_info);
        _sub_reshape_k_states.allocator()->init(*_reshape_k_states.allocator(), _sub_reshape_qkv_info);
        _sub_reshape_v_states.allocator()->init(*_reshape_v_states.allocator(), _sub_reshape_qkv_info);

        _sub_concat_reshape_kv_info.set_tensor_shape(BITensorShape(64, 12, _seq_len, batch_size));
        _sub_concat_reshape_k_states.allocator()->init(*_concat_reshape_k_states.allocator(),
                                                       _sub_concat_reshape_kv_info);
        _sub_concat_reshape_v_states.allocator()->init(*_concat_reshape_v_states.allocator(),
                                                       _sub_concat_reshape_kv_info);

        _sub_transpose_q_info.set_tensor_shape(BITensorShape(64, 1, 12, _batch_size));
        _sub_transpose_q_states.allocator()->init(*_transpose_q_states.allocator(), _sub_transpose_q_info);

        _sub_transpose_k_info.set_tensor_shape(BITensorShape(_seq_len, 64, 12, _batch_size));
        _sub_transpose_k_states.allocator()->init(*_transpose_k_states.allocator(), _sub_transpose_k_info);

        _sub_transpose_v_info.set_tensor_shape(BITensorShape(64, _seq_len, 12, _batch_size));
        _sub_transpose_v_states.allocator()->init(*_transpose_v_states.allocator(), _sub_transpose_v_info);

        _sub_qk_bmm_output_info.set_tensor_shape(BITensorShape(_seq_len, 1, 12, _batch_size));
        _sub_qk_bmm_output.allocator()->init(*_qk_bmm_output.allocator(), _sub_qk_bmm_output_info);

        _sub_softmax_output_info.set_tensor_shape(BITensorShape(_seq_len, 1, 12, _batch_size));
        _sub_softmax_output.allocator()->init(*_softmax_output.allocator(), _sub_softmax_output_info);

        _sub_pv_bmm_output_info.set_tensor_shape(BITensorShape(64, 1, 12, _batch_size));
        _sub_pv_bmm_output.allocator()->init(*_pv_bmm_output.allocator(), _sub_pv_bmm_output_info);

        _sub_pv_transpose_output_info.set_tensor_shape(BITensorShape(64, 12, 1, _batch_size));
        _sub_pv_perm_output.allocator()->init(*_pv_perm_output.allocator(), _sub_pv_transpose_output_info);

        _sub_pv_reshape_output_info.set_tensor_shape(BITensorShape(768, 1, _batch_size));
        _sub_pv_reshape_output.allocator()->init(*_pv_reshape_output.allocator(), _sub_pv_reshape_output_info);

        _sub_attn_o_output_info.set_tensor_shape(BITensorShape(768, 1, _batch_size));
        _sub_attn_o_output.allocator()->init(*_attn_o_output.allocator(), _sub_attn_o_output_info);

        std::vector<BIITensor *> outputs = {
            &_sub_query_states,
            &_sub_key_states,
            &_sub_value_states
        };

        _rms_norm_layer.dynamic_configure(input);
        _c_attn_layer.dynamic_configure();
        _split_layer.dynamic_configure(&_sub_c_attn_output, outputs);
        _reshape_q_layer.dynamic_configure();
        _reshape_k_layer.dynamic_configure();
        _reshape_v_layer.dynamic_configure();
        _transpose_q_layer.dynamic_configure(&_sub_reshape_q_states, &_sub_transpose_q_states);
        _transpose_k_layer.dynamic_configure(&_sub_concat_reshape_k_states, &_sub_transpose_k_states);
        _transpose_v_layer.dynamic_configure(&_sub_concat_reshape_v_states, &_sub_transpose_v_states);
        BIMatMulInfo matmul_info; // No transpose for lhs or rhs
        matmul_info.adj_lhs(false).adj_rhs(false);
        // Define CpuMatMulSettings
        BICpuMatMulSettings settings;
        // Enable fast math for optimization
        settings = settings.fast_math(true);
        _sub_transpose_q_states.info()->set_are_values_constant(false);
        _sub_transpose_k_states.info()->set_are_values_constant(false);
        _qk_bmm_layer.dynamic_configure(&_sub_transpose_q_states, &_sub_transpose_k_states, &_sub_qk_bmm_output);
        // _qk_add_layer.dynamic_configure(&_sub_qk_bmm_output, &_add_weights, true);
        _softmax_layer.dynamic_configure();
        _pv_bmm_layer.dynamic_configure(&_sub_softmax_output, &_sub_transpose_v_states, &_sub_pv_bmm_output);
        _pv_transpose_layer.dynamic_configure(&_sub_pv_bmm_output, &_sub_pv_perm_output);
        _pv_reshape_layer.dynamic_configure();
        _attn_o_gemm_layer.dynamic_configure();
        _c_copy_layer.dynamic_configure();
    }

    void BINEAttentionLayer::configure(BIITensor *input,
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
                                       BIITensor *output) {
        // 结果判断
        BI_COMPUTE_ERROR_ON_NULLPTR(input, gamma_weights, c_attn_bias, c_attn_weights, output); // 输入的参数是否为空
        // BI_COMPUTE_ERROR_THROW_ON(BINEAttentionLayer::validate(input->info(), weights->info(),
        //     bias->info(), output->info())); // 验证输入, 权重，偏置和输出信息
        BI_COMPUTE_LOG_PARAMS(input, gamma_weights, c_attn_weights, output); // 获取log的参数

        // 配置私有参数
        _max_seq_len = max_seq_len; // 最大的值
        _hidden_size = hidden_size; // 隐藏层长度
        _max_batch_size = batch_size; // 最大块
        _is_prepared = false; // 初始化标志，标识尚未准备好

        // 配置最大的张量信息
        const auto rms_norm_shape = BITensorShape(_hidden_size, 1, _max_batch_size); // rms norm层
        const auto c_attn_shape = BITensorShape(_hidden_size * 3, 1, _max_batch_size); // c_attn gemm的输出
        const auto reshape_qkv_shape = BITensorShape(64, 12, 1, _max_batch_size);
        const auto concat_reshape_kv_shape = BITensorShape(64, 12, _max_seq_len, _max_batch_size);
        const auto transpose_q_shape = BITensorShape(64, 1, 12, _max_batch_size);
        const auto transpose_v_shape = BITensorShape(64, _max_seq_len, 12, _max_batch_size);
        const auto transpose_k_shape = BITensorShape(_max_seq_len, 64, 12, _max_batch_size);
        auto qk_bmm_output_shape = BITensorShape(_max_seq_len, 1, 12, _max_batch_size);

        _norm_output.allocator()->init(BITensorInfo(rms_norm_shape, 1, BIDataType::F16));
        _c_attn_output.allocator()->init(BITensorInfo(c_attn_shape, 1, BIDataType::F16));
        _query_states.allocator()->init(BITensorInfo(rms_norm_shape,
                                                     1,
                                                     BIDataType::F16));
        _key_states.allocator()->init(BITensorInfo(rms_norm_shape,
                                                   1,
                                                   BIDataType::F16));
        _value_states.allocator()->init(BITensorInfo(rms_norm_shape,
                                                     1,
                                                     BIDataType::F16));
        _reshape_k_states.allocator()->init(BITensorInfo(reshape_qkv_shape, 1, BIDataType::F16));
        _reshape_q_states.allocator()->init(BITensorInfo(reshape_qkv_shape, 1, BIDataType::F16));
        _reshape_v_states.allocator()->init(BITensorInfo(reshape_qkv_shape, 1, BIDataType::F16));
        _concat_reshape_k_states.allocator()->init(BITensorInfo(concat_reshape_kv_shape, 1, BIDataType::F16));
        _concat_reshape_v_states.allocator()->init(BITensorInfo(concat_reshape_kv_shape, 1, BIDataType::F16));
        _transpose_q_states.allocator()->init(BITensorInfo(transpose_q_shape, 1, BIDataType::F16));
        _transpose_k_states.allocator()->init(BITensorInfo(transpose_k_shape, 1, BIDataType::F16));
        _transpose_v_states.allocator()->init(BITensorInfo(transpose_v_shape, 1, BIDataType::F16));
        _qk_bmm_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        _softmax_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        _pv_bmm_output.allocator()->init(BITensorInfo(transpose_q_shape, 1, BIDataType::F16));
        _pv_perm_output.allocator()->init(BITensorInfo(reshape_qkv_shape, 1, BIDataType::F16));
        _pv_reshape_output.allocator()->init(BITensorInfo(rms_norm_shape, 1, BIDataType::F16));
        _attn_o_output.allocator()->init(BITensorInfo(rms_norm_shape, 1, BIDataType::F16));

        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_c_attn_output);
        _memory_group.manage(&_query_states);
        _memory_group.manage(&_key_states);
        _memory_group.manage(&_value_states);
        _memory_group.manage(&_reshape_k_states);
        _memory_group.manage(&_reshape_v_states);
        _memory_group.manage(&_reshape_q_states);
        _memory_group.manage(&_concat_reshape_k_states);
        _memory_group.manage(&_concat_reshape_v_states);
        _memory_group.manage(&_transpose_k_states);
        _memory_group.manage(&_transpose_v_states);
        _memory_group.manage(&_transpose_q_states);
        _memory_group.manage(&_qk_bmm_output);
        _memory_group.manage(&_softmax_output);
        _memory_group.manage(&_pv_bmm_output);
        _memory_group.manage(&_pv_perm_output);
        _memory_group.manage(&_pv_reshape_output);
        _memory_group.manage(&_attn_o_output);

        _norm_output.allocator()->allocate();
        _c_attn_output.allocator()->allocate();
        _query_states.allocator()->allocate();
        _key_states.allocator()->allocate();
        _value_states.allocator()->allocate();
        _reshape_k_states.allocator()->allocate();
        _reshape_v_states.allocator()->allocate();
        _reshape_q_states.allocator()->allocate();
        _concat_reshape_k_states.allocator()->allocate();
        _concat_reshape_v_states.allocator()->allocate();
        _transpose_k_states.allocator()->allocate();
        _transpose_v_states.allocator()->allocate();
        _transpose_q_states.allocator()->allocate();
        _qk_bmm_output.allocator()->allocate();
        _softmax_output.allocator()->allocate();
        _pv_bmm_output.allocator()->allocate();
        _pv_perm_output.allocator()->allocate();
        _pv_reshape_output.allocator()->allocate();
        _attn_o_output.allocator()->allocate();

        // Sub张量初始化
        const auto _sub_norm_shape = BITensorShape(_hidden_size, 1, _batch_size);
        _sub_norm_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_norm_info.set_format(Format::F16);
        _sub_norm_output.allocator()->init(_sub_norm_info);

        const auto _sub_c_attn_shape = BITensorShape(_hidden_size * 3, 1, _batch_size);
        _sub_c_attn_tensor_info = BITensorInfo(_sub_c_attn_shape, 1, BIDataType::F16);
        _sub_c_attn_tensor_info.set_format(Format::F16);
        _sub_c_attn_output.allocator()->init(_sub_c_attn_tensor_info);

        _sub_qkv_states_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_qkv_states_info.set_format(Format::F16);
        _sub_query_states.allocator()->init(_sub_qkv_states_info);
        _sub_value_states.allocator()->init(_sub_qkv_states_info);
        _sub_key_states.allocator()->init(_sub_qkv_states_info);

        const auto sub_qkv_reshape = BITensorShape(64, 12, 1, _batch_size);
        _sub_reshape_qkv_info = BITensorInfo(sub_qkv_reshape, 1, BIDataType::F16);
        _sub_reshape_qkv_info.set_format(Format::F16);
        _sub_reshape_q_states.allocator()->init(_sub_reshape_qkv_info);
        _sub_reshape_v_states.allocator()->init(_sub_reshape_qkv_info);
        _sub_reshape_k_states.allocator()->init(_sub_reshape_qkv_info);

        const auto sub_concat_qkv_reshape = BITensorShape(64, 12, _seq_len, _batch_size);
        _sub_concat_reshape_kv_info = BITensorInfo(sub_concat_qkv_reshape, 1, BIDataType::F16);
        _sub_concat_reshape_kv_info.set_format(Format::F16);;
        _sub_concat_reshape_k_states.allocator()->init(_sub_concat_reshape_kv_info);
        _sub_concat_reshape_v_states.allocator()->init(_sub_concat_reshape_kv_info);

        const auto sub_transpose_q_shape = BITensorShape(
            64, 1, 12, _batch_size);
        _sub_transpose_q_info = BITensorInfo(sub_transpose_q_shape, 1, BIDataType::F16);
        _sub_transpose_q_info.set_format(Format::F16);
        _sub_transpose_q_states.allocator()->init(_sub_transpose_q_info);

        const auto sub_transpose_k_shape = BITensorShape(_seq_len, 64, 12, _batch_size);
        _sub_transpose_k_info = BITensorInfo(sub_transpose_k_shape, 1, BIDataType::F16);
        _sub_transpose_k_info.set_format(Format::F16);
        _sub_transpose_k_states.allocator()->init(_sub_transpose_k_info);

        const auto sub_transpose_v_shape = BITensorShape(
            64, _seq_len, 12, _batch_size);

        _sub_transpose_v_info = BITensorInfo(sub_transpose_v_shape, 1, BIDataType::F16);
        _sub_transpose_v_info.set_format(Format::F16);
        _sub_transpose_v_states.allocator()->init(_sub_transpose_v_info);

        const auto sub_qk_bmm_output_shape = BITensorShape(_seq_len, 1, 12, _batch_size);
        _sub_qk_bmm_output_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::F16);
        _sub_qk_bmm_output_info.set_format(Format::F16);
        _sub_qk_bmm_output.allocator()->init(_sub_qk_bmm_output_info);

        _sub_softmax_output_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::F16);
        _sub_softmax_output_info.set_format(Format::F16);
        _sub_softmax_output.allocator()->init(_sub_softmax_output_info);

        _sub_pv_bmm_output_info = BITensorInfo(sub_transpose_q_shape, 1, BIDataType::F16);
        _sub_pv_bmm_output_info.set_format(Format::F16);
        _sub_pv_bmm_output.allocator()->init(_sub_pv_bmm_output_info);

        _sub_pv_transpose_output_info = BITensorInfo(sub_qkv_reshape, 1, BIDataType::F16);
        _sub_pv_transpose_output_info.set_format(Format::F16);
        _sub_pv_perm_output.allocator()->init(_sub_pv_transpose_output_info);

        _sub_pv_reshape_output_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_pv_reshape_output_info.set_format(Format::F16);
        _sub_pv_reshape_output.allocator()->init(_sub_pv_reshape_output_info);

        _sub_attn_o_output_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_attn_o_output_info.set_format(Format::F16);
        _sub_attn_o_output.allocator()->init(_sub_attn_o_output_info);


        // 配置层的效果
        _rms_norm_layer.configure(input, gamma_weights, &_sub_norm_output);
        // _rms_norm_layer.configure(input, gamma_weights, &_sub_norm_output);
        GEMMInfo gemm_info;
        gemm_info.set_fast_math(true);
        _c_attn_layer.configure(&_sub_norm_output, c_attn_weights, c_attn_bias, &_sub_c_attn_output, 1.0f, 1.0f,
                                gemm_info);
        std::vector<BIITensor *> outputs = {&_sub_query_states, &_sub_key_states, &_sub_value_states};
        _split_layer.configure(&_sub_c_attn_output, outputs, 0);
        _reshape_q_layer.configure(&_sub_query_states, &_sub_reshape_q_states);
        _reshape_k_layer.configure(&_sub_key_states, &_sub_reshape_k_states);
        _reshape_v_layer.configure(&_sub_value_states, &_sub_reshape_v_states);
        _transpose_q_layer.configure(&_sub_reshape_q_states, &_sub_transpose_q_states, q_perm);
        _transpose_k_layer.configure(&_sub_concat_reshape_k_states, &_sub_transpose_k_states, k_perm);
        _transpose_v_layer.configure(&_sub_concat_reshape_v_states, &_sub_transpose_v_states, q_perm);
        _sub_transpose_q_states.info()->set_are_values_constant(false);
        _sub_transpose_k_states.info()->set_are_values_constant(false);
        BIMatMulInfo matmul_info; // No transpose for lhs or rhs
        matmul_info.adj_lhs(false).adj_rhs(false);
        // Define CpuMatMulSettings
        BICpuMatMulSettings settings;
        // Enable fast math for optimization
        // settings = settings.fast_math(true);
        _qk_bmm_layer.configure(&_sub_transpose_q_states, &_sub_transpose_k_states, &_sub_qk_bmm_output, matmul_info,
                                settings);
        _softmax_layer.configure(&_sub_qk_bmm_output, &_sub_softmax_output);
        _sub_softmax_output.info()->set_are_values_constant(false);
        _sub_transpose_v_states.info()->set_are_values_constant(false);
        _pv_bmm_layer.configure(&_sub_softmax_output, &_sub_transpose_v_states, &_sub_pv_bmm_output, matmul_info,
                                settings);
        _pv_transpose_layer.configure(&_sub_pv_bmm_output, &_sub_pv_perm_output, qkv_perm);
        _pv_reshape_layer.configure(&_sub_pv_perm_output, &_sub_pv_reshape_output);
        _attn_o_gemm_layer.configure(&_sub_pv_reshape_output, o_attn_weights, o_attn_bias, &_sub_attn_o_output, 1.0f,
                                     1.0f,
                                     gemm_info);
        _c_copy_layer.configure(&_sub_attn_o_output, output);
    }

    void BINEAttentionLayer::run() {
        BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = true; // 对齐列
        prepare();

        // 执行函数
        _rms_norm_layer.run(); // 归一化 layer norm
        _c_attn_layer.run();
        _split_layer.run();
        _reshape_q_layer.run();
        _reshape_k_layer.run();
        _reshape_v_layer.run();
        store_kv_cache();
        concat_kv_cache();

        _transpose_q_layer.run();
        _transpose_k_layer.run();
        _transpose_v_layer.run();
        _qk_bmm_layer.run(); // 计算add之前先给add_weights值进行修改
        _softmax_layer.run();
        _pv_bmm_layer.run();
        _pv_transpose_layer.run();
        _pv_reshape_layer.run();
        _attn_o_gemm_layer.run();
        _c_copy_layer.run();
        _sub_pv_reshape_output.print(std::cout, format);
    }

    void BINEAttentionLayer::prepare() {
        if (!_is_prepared) {
            // 1. 先调用内存管理组(再进行sub tensor的内存分布, 申请开辟连续内存)
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_info);
            _sub_c_attn_output.allocator()->init(*_c_attn_output.allocator(), _sub_c_attn_tensor_info);
            _sub_query_states.allocator()->init(*_query_states.allocator(), _sub_qkv_states_info);
            _sub_key_states.allocator()->init(*_key_states.allocator(), _sub_qkv_states_info);
            _sub_value_states.allocator()->init(*_value_states.allocator(), _sub_qkv_states_info);
            _sub_reshape_q_states.allocator()->init(*_reshape_q_states.allocator(), _sub_reshape_qkv_info);
            _sub_reshape_k_states.allocator()->init(*_reshape_k_states.allocator(), _sub_reshape_qkv_info);
            _sub_reshape_v_states.allocator()->init(*_reshape_v_states.allocator(), _sub_reshape_qkv_info);
            _sub_concat_reshape_k_states.allocator()->
                    init(*_concat_reshape_k_states.allocator(), _sub_concat_reshape_kv_info);
            _sub_concat_reshape_v_states.allocator()->
                    init(*_concat_reshape_v_states.allocator(), _sub_concat_reshape_kv_info);
            _sub_transpose_q_states.allocator()->init(*_transpose_q_states.allocator(), _sub_transpose_q_info);
            _sub_transpose_k_states.allocator()->init(*_transpose_k_states.allocator(), _sub_transpose_k_info);
            _sub_transpose_v_states.allocator()->init(*_transpose_v_states.allocator(), _sub_transpose_v_info);
            _sub_qk_bmm_output.allocator()->init(*_qk_bmm_output.allocator(), _sub_qk_bmm_output_info);
            _sub_softmax_output.allocator()->init(*_softmax_output.allocator(), _sub_softmax_output_info);
            _sub_pv_bmm_output.allocator()->init(*_pv_bmm_output.allocator(), _sub_pv_bmm_output_info);
            _sub_pv_perm_output.allocator()->init(*_pv_perm_output.allocator(), _sub_pv_transpose_output_info);
            _sub_pv_reshape_output.allocator()->init(*_pv_reshape_output.allocator(), _sub_pv_reshape_output_info);
            _sub_attn_o_output.allocator()->init(*_attn_o_output.allocator(), _sub_attn_o_output_info);
            _is_prepared = true;
        }
    }

    void BINEAttentionLayer::set_sequence_length(int seq_len) {
    }

    void BINEAttentionLayer::get_kv_block_ids(std::vector<unsigned int> &kv_block_ids) {
        kv_block_ids = std::move(_block_ids);
    }


    void BINEAttentionLayer::store_kv_cache() {
        _block_ids.clear();
        // 如果首次的话只会存入<s>的首字符
        if (_is_first_kv_cache) {
            const auto root_id = KVCacheManager::getInstance().root_id();
            KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_k_states.buffer(), root_id, 0, true);
            KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_v_states.buffer(), root_id, 0);

            _is_first_kv_cache = false;
            _block_ids.emplace_back(root_id);
            return;
        }
        // 判断当前的batch_size, 先根据batch size分配一组block_id
        auto _batch_index = 0;
        for (const auto &decode_list: _kv_decode_ids) {
            auto block_ids = KVCacheManager::getInstance().alloc_decode_next(
                decode_list[0], decode_list.size() - 1, decode_list);
            // 进行内存值拷贝
            for (const auto &block_id: block_ids) {
                KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_k_states.buffer(), block_id, _batch_index, true);
                KVCacheManager::getInstance().memcpy_decode_buffer(_sub_reshape_v_states.buffer(), block_id, _batch_index);
                _block_ids.emplace_back(block_id);
            }
            _batch_index++;
        }
    }

    void BINEAttentionLayer::concat_kv_cache() {
        // if (_seq_len <= 1) {
        //     return;
        // }
        // 1. 根据前面的叶子节点,先进行合并
        // TODO: 后续根据Sequence进行动态reserve
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
        // BIIOFormatInfo format;
        // format.element_delim = ", "; // 元素之间用逗号分隔
        // format.row_delim = "\n"; // 每行换行
        // format.align_columns = true; // 对齐列

        // if (_seq_len == 9) {
        //     std::cout << "打印Attention" << std::endl;
        //     _sub_concat_reshape_v_states.print(std::cout, format);
        // }

        // // _sub_reshape_k_states.print(std::cout, format);
        // _sub_concat_reshape_k_states.print(std::cout, format);
        // std::cout << "================================================================" << std::endl;
        // _sub_concat_reshape_v_states.print(std::cout, format);
    }
}
