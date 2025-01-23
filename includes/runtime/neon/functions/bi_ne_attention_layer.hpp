//
// Created by Mason on 2025/1/23.
//

#pragma once

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BINEAttentionLayer : public BIIFunction {
    public:
        BINEAttentionLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEAttentionLayer() : BINEAttentionLayer(BIMemoryManagerOnDemand::make_default()) {

        }

        BINEAttentionLayer(const BINEAttentionLayer &) = delete;

        BINEAttentionLayer(BINEAttentionLayer &&) = delete;

        BINEAttentionLayer &operator=(const BINEAttentionLayer &) = delete;

        BINEAttentionLayer &operator=(BINEAttentionLayer &&) = delete;

        ~BINEAttentionLayer();

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
        * @param weights 权重张量，形状为 [input_size, num_units]. Data types supported: Same as @p input
        * @param recurrent_weights 循环权重张量，形状为 [num_units, num_units]
        * @param bias 偏置向量，形状为 [num_units]
        * @param hidden_state 隐藏状态张量，形状为 [num_units, batch_size]
        * @param output 输出张量，形状为 [num_units, batch_size]
        * @param info 激活层参数，用于定义激活函数（如 ReLU、Tanh 等）
        */
        void configure(const BIITensor *input,
                       const BIITensor *weights,
                       const BIITensor *bias,
                       BIITensor *output);

        /**
         * 验证函数
         * @param input
         * @param weights
         * @param recurrent_weights
         * @param bias
         * @param hidden_state
         * @param output
         * @param info
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
        // 执行矩阵乘法(GEMM), 用于计算当前状态或隐藏层的现象变换
        BINEGEMM _gemm_state_f;
        // 执行Reshape操作
        BINEReshapeLayer _reshape;
        // 中间变量
        BITensor _gemm_output;
        // 转换变量
        BITensor _reshape_output;
        // 是否已经完全初始化
        bool _is_prepared;
    };
}