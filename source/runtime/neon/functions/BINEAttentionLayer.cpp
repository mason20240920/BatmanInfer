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

namespace BatmanInfer {
    BINEAttentionLayer::~BINEAttentionLayer() = default;

    BINEAttentionLayer::BINEAttentionLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)),
        _normalization_layer(),
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

        // _gemm_state_f.configure(&_norm_output, weights, bias, &_gemm_output, 1.0f, 1.0f, gemm_info);
        // std::vector<BIITensor *> outputs = {&_split_result_1, &_split_result_2, &_split_result_0};
        // _split_layer.configure(&_gemm_output, outputs, 0);
        // _reshape_1_f.configure(&_split_result_0, &_reshape_1_output);
        // _transpose_1_f.configure(&_reshape_1_output, &_transpose_1_output, perm);
        // _reshape_2_f.configure(&_split_result_1, &_reshape_2_output);
        // _transpose_2_f.configure(&_reshape_2_output, &_transpose_2_output, perm2);
        // _mul_1_f.configure(&_transpose_2_output,
        //                    scalar,
        //                    &_mul_1_output,
        //                    1.0f,
        //                    BIConvertPolicy::WRAP,
        //                    BIRoundingPolicy::TO_ZERO);
        // _reshape_3_f.configure(&_split_result_2, &_reshape_3_output);
        // _transpose_3_f.configure(&_reshape_3_output, &_transpose_3_output, perm);
        // _mul_2_f.configure(&_transpose_3_output,
        //                    scalar,
        //                    &_mul_2_output,
        //                    1.0f,
        //                    BIConvertPolicy::WRAP,
        //                    BIRoundingPolicy::TO_ZERO);
        //
        // // Define MatMulInfo
        // BIMatMulInfo matmul_info; // No transpose for lhs or rhs
        //
        // // Define CpuMatMulSettings
        // BICpuMatMulSettings settings;
        // // Enable fast math for optimization
        // settings = settings.fast_math(true);
        // // 设置不是常量
        // _mul_1_output.info()->set_are_values_constant(false);
        // _mul_2_output.info()->set_are_values_constant(false);
        // _matmul_1_f.configure(&_mul_2_output,
        //                       &_mul_1_output,
        //                       &_matmul_1_output, matmul_info, settings);
        // _add_f.configure(&_matmul_1_output,
        //                  add_weights,
        //                  &_add_output,
        //                  BIConvertPolicy::SATURATE);
        // _softmax_layer.configure(&_add_output,
        //                          &_softmax_output,
        //                          1.0f,
        //                          0);
        // _transpose_1_output.info()->set_are_values_constant(false);
        // _softmax_output.info()->set_are_values_constant(false);
        // _matmul_2_f.configure(&_softmax_output,
        //                       &_transpose_1_output,
        //                       &_matmul_2_output, matmul_info, settings);
        // _transpose_final_f.configure(&_matmul_2_output, &_transpose_final_output, final_perm);
        // _reshape_final_f.configure(&_transpose_final_output, &_reshape_final_output);
        // _gemm_final_f.configure(&_reshape_final_output, weights_second, bias_second, &_gemm_final_output, 1.f, 1.f,
        //                         gemm_info);
        // _copy_f.configure(&_gemm_final_output, output);
    }

    void BINEAttentionLayer::run() {
        // 输入格式
        BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列
        //        prepare();

        BIMemoryGroupResourceScope scope_mg(_memory_group);

        // 执行函数
        _normalization_layer.run(); // 归一化 layer norm
        // _gemm_state_f.run();
        // _split_layer.run();
        // _reshape_1_f.run();
        // _transpose_1_f.run();
        // _reshape_2_f.run();
        // _transpose_2_f.run();
        // _mul_1_f.run();
        // _reshape_3_f.run();
        // _transpose_3_f.run();
        // _mul_2_f.run();
        // _matmul_1_f.run();
        // _add_f.run();
        // _softmax_layer.run();
        // _matmul_2_f.run();
        // _transpose_final_f.run();
        // _reshape_final_f.run();
        // _gemm_final_f.run();
        // _copy_f.run(); // 运行拷贝
    }

    void BINEAttentionLayer::prepare() {
        if (!_is_prepared) {
            //            _reshape.prepare();
            //            _gemm_state_f.prepare();

            _is_prepared = true;
        }
    }

    void BINEAttentionLayer::set_sequence_length(int seq_len) {
    }
}
