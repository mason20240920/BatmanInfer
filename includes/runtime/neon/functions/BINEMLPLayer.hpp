//
// Created by Mason on 2025/4/7.
//

#pragma once

#include <runtime/bi_i_function.hpp>

#include "BINERMSNormLayer.hpp"
#include "bi_NEDequantizationLayer.h"
#include "bi_NEQuantizationLayer.h"
#include "bi_ne_copy.hpp"
#include "bi_ne_gemm_lowp_matrix_mul_core.hpp"
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
         */
        void configure(const BIITensor *input,
                       const BIITensor *fc_weights,
                       const BIITensor *fc_bias,
                       const BIITensor *proj_weights,
                       const BIITensor *proj_bias,
                       const BIITensor *gamma,
                       const BIActivationLayerInfo &act_info,
                       BIITensor *output,
                       const size_t &batch_size,
                       const size_t &seq_len);

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *fc_weights,
                                 const BIITensorInfo *fc_bias,
                                 const BIITensorInfo *proj_weights,
                                 const BIITensorInfo *proj_bias,
                                 const BIITensorInfo *gamma,
                                 const BIITensorInfo *output);

        void run() override;

    private:
        // 算子操作
        BINERMSNormLayer _rms_layer; // 用于执行归一操作的层

        // BINEQuantizationLayer _quantization_layer; // 量化操作, 将数据量化为int8
        //
        // BINEGEMMLowpMatrixMultipleCore _matrix_mul_core; // 量化的Core操作
        //
        // BINEGEMMLowpMatrixMultipleCore _c_proj; // 扩展维度
        //
        // BINEDequantizationLayer _dequantization_layer; // 反量化
        //
        // BINECopy _copy_f; // 拷贝张量操作

    private:
        // 张量信息
        BIMemoryGroup _memory_group; // 内存管理

        BITensor _norm_output, _norm_q_output;
        // BITensor _fuse_output;
        // BITensor _proj_output, _proj_q_output;

        // 参数长度
        size_t _max_batch;
        size_t _max_seq;
    };
} // namespace BatmanInfer
