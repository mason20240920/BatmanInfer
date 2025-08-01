//
// Created by Mason on 2025/1/23.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/neon/functions/bi_NEActivationLayer.h>
#include <runtime/neon/functions/bi_NEArithmeticAddition.h>
#include <runtime/neon/functions/bi_ne_copy.hpp>
#include <runtime/neon/functions/bi_NEFullyConnectedLayer.h>
#include <runtime/neon/functions/bi_ne_gemm.hpp>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    /**
     * 基础算子: 运行 @ref BINERNNLayer
     */
    class BINERNNLayer : public BIIFunction {
    public:
        BINERNNLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINERNNLayer() : BINERNNLayer(BIMemoryManagerOnDemand::make_default()) {

        }

        BINERNNLayer(const BINERNNLayer &) = delete;

        BINERNNLayer(BINERNNLayer &&) = delete;

        BINERNNLayer &operator=(const BINERNNLayer &) = delete;

        BINERNNLayer &operator=(BINERNNLayer &&) = delete;

        ~BINERNNLayer();

        /**
         * 初始化函数
         * Valid data layouts:
         * - NHWC
         * - NCHW
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
                       const BIITensor *recurrent_weights,
                       const BIITensor *bias,
                       BIITensor *hidden_state,
                       BIITensor *output,
                       BIActivationLayerInfo &info);

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
                                 const BIITensorInfo *recurrent_weights,
                                 const BIITensorInfo *bias,
                                 const BIITensorInfo *hidden_state,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &info);

        // 继承函数
        void run();

        void prepare() override;

    private:
        // 内存管理
        BIMemoryGroup _memory_group;
        // 执行矩阵乘法(GEMM)对象,用于计算当前状态或隐藏层的线性变换
        BINEGEMM _gemm_state_f;
        // 执行加法操作
        BINEArithmeticAddition _add_f;
        // 激活函数对象
        BINEActivationLayer _activation;
        // 全连接层
        BINEFullyConnectedLayer _fully_connected;
        // 复制的对象
        BINECopy _copy_f;
        // 中间变量
        BITensor _fully_connected_out;
        BITensor _gemm_output;
        BITensor _add_output;
        // 层是否已经完全初始化
        bool _is_prepared;
    };
}