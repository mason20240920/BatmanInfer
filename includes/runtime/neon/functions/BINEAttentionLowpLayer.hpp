//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include "BINERMSNormLayer.hpp"
#include "bi_NEArithmeticAddition.h"
#include "bi_NEDequantizationLayer.h"
#include "bi_NEQuantizationLayer.h"
#include "bi_NESoftmaxLayer.h"
#include "bi_ne_gemm_lowp_matrix_mul_core.hpp"
#include "bi_ne_gemm_lowp_output_stage.hpp"
#include "bi_ne_mat_mul.hpp"
#include "bi_ne_permute.h"
#include "bi_ne_split.hpp"
#include "bi_ne_transpose.hpp"

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BINEAttentionLowpLayer : public BIIFunction {
    public:
        explicit BINEAttentionLowpLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEAttentionLowpLayer() : BINEAttentionLowpLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEAttentionLowpLayer(const BINEAttentionLowpLayer &) = delete;

        BINEAttentionLowpLayer(BINEAttentionLowpLayer &&) = delete;

        BINEAttentionLowpLayer &operator=(const BINEAttentionLowpLayer &) = delete;

        BINEAttentionLowpLayer &operator=(BINEAttentionLowpLayer &&) = delete;

        ~BINEAttentionLowpLayer() override;

        /**
        * 初始化函数
        * Valid data layouts:
        * - NHWC
        *
        * Valid data type configurations:
        * |src0   |src1   |src2   |src3   |dst0   |dst1   |
        * |:------|:------|:------|:------|:------|:------|
        * |F16    |F16    |F16    |F16    |F16    |F16    |
        * |F32    |F32    |F32    |F32    |F32    |F32    |
        * @param input 输入张量，形状为 [input_size, batch_size]。. Data types supported: F16/F32
        * @param gamma_weights
        * @param c_attn_weights 第一层注意力权重
        * @param c_attn_bias 第二层注意力权重
        * @param gemm_i_scale Gemm输入的张量量化scale
        * @param gemm_i_zp Gemm输入张量的零点
        * @param attn_gemm_o_scale
        * @param attn_gemm_o_zp
        * @param query_q_scale
        * @param query_q_zp
        * @param value_q_scale
        * @param value_q_zp
        * @param key_q_scale
        * @param key_q_zp
        * @param q_perm
        * @param hidden_size
        * @param max_seq_len
        * @param batch_size
        * @param output 输出张量，形状为 [num_units, batch_size]
        */
        void configure(const BIITensor *input,
                       const BIITensor *gamma_weights,
                       const BIITensor *c_attn_weights,
                       const BIITensor *c_attn_bias,
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
                       BIITensor *output);

        void dynamic_configure(const BIITensor *input,
                               const size_t &seq_len,
                               const size_t &batch_size);

        /**
         * 验证函数
         * @param input
         * @param weights
         * @param bias
         * @param output
         * @return
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *weights,
                                 const BIITensorInfo *bias,
                                 const BIITensorInfo *output);

        void run() override;

        void prepare() override;

    private:
        // 内存管理
        BIMemoryGroup _memory_group;

    private:
        // Attention模块算子
        BINERMSNormLayer _rms_norm_layer; // 归一化层
        BINEQuantizationLayer _quantization_layer; // 归一结果进行量化
        BINEGEMMLowpMatrixMultipleCore _c_attn_layer; // 进行channel-wise计算
        BINEGEMMLowpOutputStage _c_attn_o_stage; // 进行计算出来的output stage结果
        BINESplit _split_layer; // 切分层
        BINEDequantizationLayer _deq_q_layer, _deq_k_layer, _deq_v_layer;
        // BINEQuantizationLayer _quant_q_layer, _quant_k_layer;
        BINEQuantizationLayer _quant_v_layer;
        BINEReshapeLayer _reshape_q_layer, _reshape_k_layer, _reshape_v_layer;
        BINEPermute _transpose_q_layer, _transpose_k_layer, _transpose_v_layer;
        BINEMatMul _qk_bmm_layer;
        BINEArithmeticAddition _qk_add_layer;
        BINESoftmaxLayer _softmax_layer;
        BINEQuantizationLayer _q_softmax_layer;
        BINEMatMul _pv_bmm_layer;
        BINEPermute _pv_transpose_layer;
        BINEReshapeLayer _pv_reshape_layer;
        BINEDequantizationLayer _pv_dequantization_layer;
        BINEGEMM _attn_o_gemm_layer;

    private:
        BITensor _sub_norm_tensor;
        BITensor _sub_norm_q_tensor;
        BITensor _sub_c_attn_s32_tensor;
        BITensor _sub_c_attn_q8_tensor;
        BITensor _sub_split_q_result_0;
        BITensor _sub_split_q_result_1;
        BITensor _sub_split_q_result_2;
        BITensor _sub_query_states;
        BITensor _sub_key_states;
        BITensor _sub_value_states;
        // BITensor _sub_q_query_states;
        BITensor _sub_q_value_states;
        // BITensor _sub_q_key_states;
        BITensor _sub_reshape_q_states;
        BITensor _sub_reshape_k_states;
        BITensor _sub_reshape_v_states;
        BITensor _sub_transpose_q_result;
        BITensor _sub_transpose_k_result;
        BITensor _sub_transpose_v_result;
        BITensor _sub_qk_bmm_output;
        BITensor _sub_add_output;
        BITensor _sub_add_weights; // Mask函数选择
        BITensor _sub_softmax_output;
        BITensor _sub_softmax_q_result;
        BITensor _sub_pv_bmm_output;
        BITensor _sub_pv_transpose_output;
        BITensor _sub_pv_reshape_output;
        BITensor _sub_pv_deq_output;
        BITensor _sub_attn_o_output;
        BITensorInfo _sub_norm_info;
        BITensorInfo _sub_norm_q_info;
        BITensorInfo _sub_c_attn_s32_tensor_info;
        BITensorInfo _sub_c_attn_q_info;
        BITensorInfo _sub_split_q_info;
        BITensorInfo _sub_d_qkv_info;
        BITensorInfo _sub_query_info;
        BITensorInfo _sub_q_query_info;
        BITensorInfo _sub_q_key_info;
        BITensorInfo _sub_q_value_info;
        BITensorInfo _sub_reshape_q_info;
        BITensorInfo _sub_reshape_k_info;
        BITensorInfo _sub_reshape_v_info;
        BITensorInfo _sub_transpose_q_info;
        BITensorInfo _sub_transpose_k_info;
        BITensorInfo _sub_transpose_v_info;
        BITensorInfo _sub_qk_bmm_output_info;
        BITensorInfo _sub_add_output_info;
        BITensorInfo _sub_add_weights_info;
        BITensorInfo _sub_softmax_output_info;
        BITensorInfo _sub_softmax_q_result_info;
        BITensorInfo _sub_pv_bmm_output_info;
        BITensorInfo _sub_pv_transpose_output_info;
        BITensorInfo _sub_pv_reshape_output_info;
        BITensorInfo _sub_pv_deq_output_info;
        BITensorInfo _sub_attn_o_output_info;

        BITensor _norm_output;
        BITensor _q_norm_output;
        BITensor _c_attn_s32_output;
        BITensor _c_attn_q8_output;
        BITensor _split_q_result_0;
        BITensor _split_q_result_1;
        BITensor _split_q_result_2;
        BITensor _query_states;
        BITensor _key_states;
        BITensor _value_states;
        // BITensor _q_query_states;
        // BITensor _q_key_states;
        BITensor _q_value_states;
        BITensor _reshape_q_states;
        BITensor _reshape_k_states;
        BITensor _reshape_v_states;
        BITensor _transpose_q_result;
        BITensor _transpose_k_result;
        BITensor _transpose_v_result;
        BITensor _qk_bmm_output;
        BITensor _add_output;
        BITensor _add_weights;
        BITensor _softmax_output;
        BITensor _q_softmax_output;
        BITensor _pv_bmm_output;
        BITensor _pv_perm_output;
        BITensor _pv_reshape_output;
        BITensor _pv_deq_output;
        BITensor _attn_o_output;

    private:
        // 是否已经完全初始化
        size_t _hidden_size{}; // 隐藏层大小
        size_t _max_seq_len{}; // 最大长度输入
        size_t _max_batch_size{}; // 一块的大小
        size_t _batch_size = 1;
        size_t _seq_len = 1;
        bool _is_prepared; // 是否已经完全初始化(预先把内存加载完)
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;
    };
}
