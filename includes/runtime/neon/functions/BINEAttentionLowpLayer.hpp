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
#include "bi_NEDequantizationLayer.h"
#include "bi_NEQuantizationLayer.h"
#include "bi_ne_gemm_lowp_matrix_mul_core.hpp"
#include "bi_ne_gemm_lowp_output_stage.hpp"
#include "bi_ne_split.hpp"

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
        BINEDequantizationLayer _deq_q_layer;
        BINEQuantizationLayer _quant_q_layer;

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
        BITensor _sub_q_query_states;
        BITensorInfo _sub_norm_info;
        BITensorInfo _sub_norm_q_info;
        BITensorInfo _sub_c_attn_s32_tensor_info;
        BITensorInfo _sub_c_attn_q_info;
        BITensorInfo _sub_split_q_info;
        BITensorInfo _sub_d_qkv_info;
        BITensorInfo _sub_query_info;
        BITensorInfo _sub_q_query_info;
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
        BITensor _q_query_states;

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
