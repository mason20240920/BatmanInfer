//
// Created by Mason on 2025/4/7.
//

#pragma once

#include <runtime/bi_i_function.hpp>

#include "BINERMSNormLayer.hpp"
#include "bi_NEActivationLayer.h"
#include "bi_NEDequantizationLayer.h"
#include "bi_NEQuantizationLayer.h"
#include "bi_ne_copy.hpp"
#include "bi_ne_gemm.hpp"
#include "bi_ne_gemm_lowp_matrix_mul_core.hpp"
#include "bi_ne_gemm_lowp_output_stage.hpp"
#include "runtime/bi_memory_group.hpp"

namespace BatmanInfer {
    class BINEMLPLayer : public BIIFunction {
    public:
        explicit BINEMLPLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEMLPLayer(): BINEMLPLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEMLPLayer(const BINEMLPLayer &) = delete;

        BINEMLPLayer(BINEMLPLayer &&) = delete;

        BINEMLPLayer &operator=(const BINEMLPLayer &) = delete;

        BINEMLPLayer &operator=(BINEMLPLayer &&) = delete;

        ~BINEMLPLayer() override;

        /**
         * @brief
         * @param input
         * @param fc_weights
         * @param fc_bias
         * @param proj_weights
         * @param proj_bias
         * @param gamma
         * @param act_info
         * @param output
         * @param max_batch_size
         * @param max_seq_len
         */
        void configure(const BIITensor *input,
                       const float fc1_input_scale,
                       const int fc1_input_zero_point,
                       const BIITensor *fc_weights,
                       const BIITensor *fc_bias,
                       const BIQuantizationInfo *c_fc_weight_qinfo,
                       const float fc1_output_scale,
                       const int fc1_output_zero_point,
                       const float gelu_output_scale,
                       const int gelu_output_zero_point,
                       const BIITensor *proj_weights,
                       const BIITensor *proj_bias,
                       const BIITensor *gamma,
                       BIITensor *output,
                       const size_t &max_batch_size,
                       const size_t &max_seq_len
        );

        void dynamic_configure(const BIITensor *input,
                               const size_t &batch_size);

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *fc_weights,
                                 const BIITensorInfo *fc_bias,
                                 const BIITensorInfo *proj_weights,
                                 const BIITensorInfo *proj_bias,
                                 const BIITensorInfo *gamma,
                                 const BIITensorInfo *output);

        void run();

        void prepare() override;

    private:
        // 将量化信息取反
        void invert_qinfo_offset(BITensor &t);

    private:
        // 算子操作
        BINERMSNormLayer _rms_layer; // 用于执行归一操作的层

        BINEQuantizationLayer _quantization_layer; // 量化操作, 将数据量化为int8

        BINEGEMMLowpMatrixMultipleCore _matrix_mul_core; // 量化的Core操作

        BINEGEMMLowpOutputStage _gemm_lowp_output_stage; // 进行S32转为量化int8

        BINEActivationLayer _activation_layer; // 激活函数

        BINEGEMM _c_proj; // 扩展维度

        BINEDequantizationLayer _dequantization_layer; // 反量化

        BINECopy _copy_f; // 拷贝张量操作

    private:
        // 张量信息
        BIMemoryGroup _memory_group; // 内存管理
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;

        BITensor _norm_output, _norm_q_output;
        BITensor _fc_q_output, _fc_s32_output;
        BITensor _act_output;
        BITensor _proj_input, _proj_output;

        BITensor _sub_norm_output, _sub_norm_q_output;
        BITensor _sub_fc_q_output, _sub_fc_s32_output;
        BITensor _sub_act_output;
        BITensor _sub_proj_input, _sub_proj_output;

        BITensorInfo _sub_norm_output_info, _sub_norm_q_output_info;
        BITensorInfo _sub_fc_q_output_info, _sub_fc_s32_output_info;
        BITensorInfo _sub_act_output_info;
        BITensorInfo _sub_proj_input_info, _sub_proj_output_info;
        bool _is_prepared{false};

        // 参数长度
        size_t _max_batch;
        size_t _max_seq;

        size_t _batch_size = 1;
        size_t _seq_len = 1;
    };
} // namespace BatmanInfer
