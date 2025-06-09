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
        _is_prepared(false) {
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

    void BINEAttentionLayer::dynamic_configure(const BIITensor *input,
                                               const size_t &seq_len,
                                               const size_t &batch_size) {
        _batch_size = batch_size;
        _seq_len = seq_len;

        _sub_norm_info.set_tensor_shape(BITensorShape(_hidden_size, seq_len, batch_size));
        _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_info);

        _sub_c_attn_tensor_info.set_tensor_shape(BITensorShape(_hidden_size * 3, seq_len, batch_size));
        _sub_c_attn_output.allocator()->init(*_c_attn_output.allocator(), _sub_c_attn_tensor_info);

        _sub_qkv_states_info.set_tensor_shape(BITensorShape(_hidden_size, seq_len, batch_size));
        _sub_query_states.allocator()->init(*_query_states.allocator(), _sub_qkv_states_info);
        _sub_key_states.allocator()->init(*_key_states.allocator(), _sub_qkv_states_info);
        _sub_value_states.allocator()->init(*_value_states.allocator(), _sub_qkv_states_info);

        _sub_reshape_qkv_info.set_tensor_shape(BITensorShape(64, 12, seq_len, batch_size));
        _sub_reshape_q_states.allocator()->init(*_reshape_q_states.allocator(), _sub_reshape_qkv_info);
        _sub_reshape_k_states.allocator()->init(*_reshape_k_states.allocator(), _sub_reshape_qkv_info);
        _sub_reshape_v_states.allocator()->init(*_reshape_v_states.allocator(), _sub_reshape_qkv_info);

        _sub_transpose_q_info.set_tensor_shape(BITensorShape(64, _seq_len, 12, _batch_size));
        _sub_transpose_q_states.allocator()->init(*_transpose_q_states.allocator(), _sub_transpose_q_info);

        _sub_transpose_k_info.set_tensor_shape(BITensorShape(_seq_len, 64, 12, _batch_size));
        _sub_transpose_k_states.allocator()->init(*_transpose_k_states.allocator(), _sub_transpose_k_info);

        _sub_transpose_v_info.set_tensor_shape(BITensorShape(64, _seq_len, 12, _batch_size));
        _sub_transpose_v_states.allocator()->init(*_transpose_v_states.allocator(), _sub_transpose_v_info);

        _sub_qk_bmm_output_info.set_tensor_shape(BITensorShape(_seq_len, _seq_len, 12, _batch_size));
        _sub_qk_bmm_output.allocator()->init(*_qk_bmm_output.allocator(), _sub_qk_bmm_output_info);

        _sub_add_weights_info.set_tensor_shape(BITensorShape(_seq_len, _seq_len));
        _sub_add_weights.allocator()->init(*_add_weights.allocator(), _sub_add_weights_info);

        _sub_add_output_info.set_tensor_shape(BITensorShape(_seq_len, _seq_len, 12, _batch_size));
        _sub_add_output.allocator()->init(*_add_output.allocator(), _sub_add_output_info);

        _sub_softmax_output_info.set_tensor_shape(BITensorShape(_seq_len, _seq_len, 12, _batch_size));
        _sub_softmax_output.allocator()->init(*_softmax_output.allocator(), _sub_softmax_output_info);

        _sub_pv_bmm_output_info.set_tensor_shape(BITensorShape(64, _seq_len, 12, _batch_size));
        _sub_pv_bmm_output.allocator()->init(*_pv_bmm_output.allocator(), _sub_pv_bmm_output_info);

        _sub_pv_transpose_output_info.set_tensor_shape(BITensorShape(64, 12, _seq_len, _batch_size));
        _sub_pv_perm_output.allocator()->init(*_pv_perm_output.allocator(), _sub_pv_transpose_output_info);

        _sub_pv_reshape_output_info.set_tensor_shape(BITensorShape(768, _seq_len, _batch_size));
        _sub_pv_reshape_output.allocator()->init(*_pv_reshape_output.allocator(), _sub_pv_reshape_output_info);

        _sub_attn_o_output_info.set_tensor_shape(BITensorShape(768, _seq_len, _batch_size));
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
        _transpose_k_layer.dynamic_configure(&_sub_reshape_k_states, &_sub_transpose_k_states);
        _transpose_v_layer.dynamic_configure(&_sub_reshape_v_states, &_sub_transpose_v_states);
        BIMatMulInfo matmul_info; // No transpose for lhs or rhs
        matmul_info.adj_lhs(false).adj_rhs(false);
        // Define CpuMatMulSettings
        BICpuMatMulSettings settings;
        // Enable fast math for optimization
        settings = settings.fast_math(true);
        _sub_transpose_q_states.info()->set_are_values_constant(false);
        _sub_transpose_k_states.info()->set_are_values_constant(false);
        _qk_bmm_layer.dynamic_configure(&_sub_transpose_q_states, &_sub_transpose_k_states, &_sub_qk_bmm_output);
        _qk_add_layer.dynamic_configure(&_sub_qk_bmm_output, &_add_weights, true);
        _softmax_layer.dynamic_configure();
        _pv_bmm_layer.dynamic_configure(&_sub_softmax_output, &_sub_transpose_v_states, &_sub_pv_bmm_output);
        _pv_transpose_layer.dynamic_configure(&_sub_pv_bmm_output, &_sub_pv_perm_output);
        _pv_reshape_layer.dynamic_configure();
        _attn_o_gemm_layer.dynamic_configure();
        _c_copy_layer.dynamic_configure();
    }

    void BINEAttentionLayer::configure(const BIITensor *input,
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
        const auto rms_norm_shape = BITensorShape(_hidden_size, _max_seq_len, _max_batch_size); // rms norm层
        const auto c_attn_shape = BITensorShape(_hidden_size * 3, _max_seq_len, _max_batch_size); // c_attn gemm的输出
        const auto reshape_qkv_shape = BITensorShape(64, 12, _max_seq_len, _max_batch_size);
        const auto transpose_q_shape = BITensorShape(64, _max_seq_len, 12, _max_batch_size);
        const auto transpose_k_shape = BITensorShape(_max_seq_len, 64, 12, _max_batch_size);
        auto qk_bmm_output_shape = BITensorShape(_max_seq_len, _max_seq_len, 12, _max_batch_size);
        const auto add_weights_shape = BITensorShape(_max_seq_len, _max_seq_len);

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
        _transpose_q_states.allocator()->init(BITensorInfo(transpose_q_shape, 1, BIDataType::F16));
        _transpose_k_states.allocator()->init(BITensorInfo(transpose_k_shape, 1, BIDataType::F16));
        _transpose_v_states.allocator()->init(BITensorInfo(transpose_q_shape, 1, BIDataType::F16));
        _qk_bmm_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        _add_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        _softmax_output.allocator()->init(BITensorInfo(qk_bmm_output_shape, 1, BIDataType::F16));
        _pv_bmm_output.allocator()->init(BITensorInfo(transpose_q_shape, 1, BIDataType::F16));
        _pv_perm_output.allocator()->init(BITensorInfo(reshape_qkv_shape, 1, BIDataType::F16));
        _pv_reshape_output.allocator()->init(BITensorInfo(rms_norm_shape, 1, BIDataType::F16));
        _attn_o_output.allocator()->init(BITensorInfo(rms_norm_shape, 1, BIDataType::F16));
        _add_weights.allocator()->init(BITensorInfo(add_weights_shape, 1, BIDataType::F16));

        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_c_attn_output);
        _memory_group.manage(&_query_states);
        _memory_group.manage(&_key_states);
        _memory_group.manage(&_value_states);
        _memory_group.manage(&_reshape_k_states);
        _memory_group.manage(&_reshape_v_states);
        _memory_group.manage(&_reshape_q_states);
        _memory_group.manage(&_transpose_k_states);
        _memory_group.manage(&_transpose_v_states);
        _memory_group.manage(&_transpose_q_states);
        _memory_group.manage(&_qk_bmm_output);
        _memory_group.manage(&_add_output);
        _memory_group.manage(&_softmax_output);
        _memory_group.manage(&_pv_bmm_output);
        _memory_group.manage(&_pv_perm_output);
        _memory_group.manage(&_pv_reshape_output);
        _memory_group.manage(&_attn_o_output);
        _memory_group.manage(&_add_weights);

        _norm_output.allocator()->allocate();
        _c_attn_output.allocator()->allocate();
        _query_states.allocator()->allocate();
        _key_states.allocator()->allocate();
        _value_states.allocator()->allocate();
        _reshape_k_states.allocator()->allocate();
        _reshape_v_states.allocator()->allocate();
        _reshape_q_states.allocator()->allocate();
        _transpose_k_states.allocator()->allocate();
        _transpose_v_states.allocator()->allocate();
        _transpose_q_states.allocator()->allocate();
        _qk_bmm_output.allocator()->allocate();
        _add_output.allocator()->allocate();
        _softmax_output.allocator()->allocate();
        _pv_bmm_output.allocator()->allocate();
        _pv_perm_output.allocator()->allocate();
        _pv_reshape_output.allocator()->allocate();
        _attn_o_output.allocator()->allocate();
        _add_weights.allocator()->allocate();

        // Sub张量初始化
        const auto _sub_norm_shape = BITensorShape(_hidden_size, _seq_len, _batch_size);
        _sub_norm_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_norm_info.set_format(Format::F16);
        _sub_norm_output.allocator()->init(_sub_norm_info);

        const auto _sub_c_attn_shape = BITensorShape(_hidden_size * 3, _seq_len, _batch_size);
        _sub_c_attn_tensor_info = BITensorInfo(_sub_c_attn_shape, 1, BIDataType::F16);
        _sub_c_attn_tensor_info.set_format(Format::F16);
        _sub_c_attn_output.allocator()->init(_sub_c_attn_tensor_info);

        _sub_qkv_states_info = BITensorInfo(_sub_norm_shape, 1, BIDataType::F16);
        _sub_qkv_states_info.set_format(Format::F16);
        _sub_query_states.allocator()->init(_sub_qkv_states_info);
        _sub_value_states.allocator()->init(_sub_qkv_states_info);
        _sub_key_states.allocator()->init(_sub_qkv_states_info);

        const auto sub_qkv_reshape = BITensorShape(64, 12, _seq_len, _batch_size);
        _sub_reshape_qkv_info = BITensorInfo(sub_qkv_reshape, 1, BIDataType::F16);
        _sub_reshape_qkv_info.set_format(Format::F16);
        _sub_reshape_q_states.allocator()->init(_sub_reshape_qkv_info);
        _sub_reshape_v_states.allocator()->init(_sub_reshape_qkv_info);
        _sub_reshape_k_states.allocator()->init(_sub_reshape_qkv_info);

        const auto sub_transpose_q_shape = BITensorShape(
            64, _seq_len, 12, _batch_size);
        _sub_transpose_q_info = BITensorInfo(sub_transpose_q_shape, 1, BIDataType::F16);
        _sub_transpose_q_info.set_format(Format::F16);
        _sub_transpose_q_states.allocator()->init(_sub_transpose_q_info);

        const auto sub_transpose_k_shape = BITensorShape(_seq_len, 64, 12, _batch_size);
        _sub_transpose_k_info = BITensorInfo(sub_transpose_k_shape, 1, BIDataType::F16);
        _sub_transpose_k_info.set_format(Format::F16);
        _sub_transpose_k_states.allocator()->init(_sub_transpose_k_info);

        _sub_transpose_v_info = BITensorInfo(sub_transpose_q_shape, 1, BIDataType::F16);
        _sub_transpose_v_info.set_format(Format::F16);
        _sub_transpose_v_states.allocator()->init(_sub_transpose_v_info);

        const auto sub_qk_bmm_output_shape = BITensorShape(_seq_len, _seq_len, 12, _batch_size);
        _sub_qk_bmm_output_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::F16);
        _sub_qk_bmm_output_info.set_format(Format::F16);
        _sub_qk_bmm_output.allocator()->init(_sub_qk_bmm_output_info);

        const auto sub_add_weight_shape = BITensorShape(_seq_len, _seq_len);
        _sub_add_weights_info = BITensorInfo(sub_add_weight_shape, 1, BIDataType::F16);
        _sub_add_weights_info.set_format(Format::F16);
        _sub_add_weights.allocator()->init(_sub_add_weights_info);

        _sub_add_output_info = BITensorInfo(sub_qk_bmm_output_shape, 1, BIDataType::F16);
        _sub_add_output_info.set_format(Format::F16);
        _sub_add_output.allocator()->init(_sub_add_output_info);

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
        _transpose_k_layer.configure(&_sub_reshape_k_states, &_sub_transpose_k_states, k_perm);
        _transpose_v_layer.configure(&_sub_reshape_v_states, &_sub_transpose_v_states, q_perm);
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
        _qk_add_layer.configure(&_sub_qk_bmm_output, &_sub_add_weights, &_sub_add_output, BIConvertPolicy::SATURATE);
        _softmax_layer.configure(&_sub_add_output, &_sub_softmax_output);
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
        prepare();
        // 执行函数
        _rms_norm_layer.run(); // 归一化 layer norm
        _c_attn_layer.run();
        _split_layer.run();
        _reshape_q_layer.run();
        _reshape_k_layer.run();
        _reshape_v_layer.run();
        _transpose_q_layer.run();
        _transpose_k_layer.run();
        _transpose_v_layer.run();
        // KVCacheManager::getInstance().memcpy_decode_buffer();
        // std::cout << _transpose_k_states.info()->total_size() << std::endl;
        // _sub_transpose_k_states.print(std::cout);
        _qk_bmm_layer.run(); // 计算add之前先给add_weights值进行修改
        BIWindow window;
        window.use_tensor_dimensions(_sub_add_weights_info.tensor_shape());
        BIIterator mask_it(&_sub_add_weights, window);
        execute_window_loop(window, [&](const BICoordinates &id) {
            auto x = id[0];
            auto y = id[1];
            *reinterpret_cast<float16_t *>(mask_it.ptr()) = (x <= y)
                                                                ? 0
                                                                : -std::numeric_limits<
                                                                    float>::infinity();
        }, mask_it);
        _qk_add_layer.run();
        _softmax_layer.run();
        _pv_bmm_layer.run();
        _pv_transpose_layer.run();
        _pv_reshape_layer.run();
        _attn_o_gemm_layer.run();
        _c_copy_layer.run();
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
            _sub_transpose_q_states.allocator()->init(*_transpose_q_states.allocator(), _sub_transpose_q_info);
            _sub_transpose_k_states.allocator()->init(*_transpose_k_states.allocator(), _sub_transpose_k_info);
            _sub_transpose_v_states.allocator()->init(*_transpose_v_states.allocator(), _sub_transpose_v_info);
            _sub_qk_bmm_output.allocator()->init(*_qk_bmm_output.allocator(), _sub_qk_bmm_output_info);
            _sub_add_weights.allocator()->init(*_add_weights.allocator(), _sub_add_weights_info);
            _sub_add_output.allocator()->init(*_add_output.allocator(), _sub_add_output_info);
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
}
