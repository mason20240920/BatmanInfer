//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/neon/functions/bi_ne_mat_mul.hpp>

#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>
#include "bi_ne_copy.hpp"

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
        * @param weights 权重张量，形状为 [input_size, num_units]. Data types supported: Same as @p input
        * @param bias 偏置向量，形状为 [num_units]
        * @param add_weights 相加权重
        * @param perm 进行切换的perm
        * @param output 输出张量，形状为 [num_units, batch_size]
        * @param info 激活层参数，用于定义激活函数（如 ReLU、Tanh 等）
        */
        void configure(const BIITensor *input,
                       const BIITensor *weights,
                       const BIITensor *bias,
                       const BIITensor *scalar,
                       const BIITensor *add_weights,
                       const BIITensor *weights_second,
                       const BIITensor *bias_second,
                       const PermutationVector &perm,
                       const PermutationVector &perm2,
                       const PermutationVector &final_perm,
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
        BINEGEMMLowpMatrixMultipleCore _gemm_state_f;
        // 执行Reshape操作
        BINEReshapeLayer _reshape;

        BINEReshapeLayer _reshape2;

        BINEReshapeLayer _reshape_split_0;

        BINEPermute _transpose_split_0;

        // 进行split 1 和 split 2分支的切换
        BINEReshapeLayer _reshape_split_1;

        BINEPermute _transpose_split_1;

        BINEPixelWiseMultiplication _mul_op_0;

        BINEReshapeLayer _reshape_split_2;

        BINEPermute _transpose_split_2;

        BINEPixelWiseMultiplication _mul_op_1;

        BINEArithmeticAddition _add_op;

        BINESoftmaxLayerGeneric<false> _softmax_layer;

        BINEMatMul _matmul_op;

        BINEMatMul _matmul_op1;

        BINEPermute _transpose_sum;

        BINEReshapeLayer _reshape_sum_layer;

        // 执行矩阵乘法(GEMM), 用于计算当前状态或隐藏层的现象变换
        BINEGEMMLowpMatrixMultipleCore _gemm_state_sum_layer;

        // 最后的Reshape Layer
        BINEReshapeLayer _final_reshape_layer;
        // 进行切分split
        BINESplit _split_layer;
        // 中间变量
        BITensor _gemm_output;
        // 转换变量
        BITensor _reshape_output;
        // 第二层转换变量
        BITensor _reshape_output_2;
        // 输出split结果
        BITensor _split_result_0;
        BITensor _split_result_1;
        BITensor _split_result_2;
        // split 0 分支的reshape
        BITensor _reshape_split_output_0;

        BITensor _transpose_split_output_0;
        // split 1 分支的代码
        BITensor _reshape_split_output_1;

        BITensor _transpose_split_output_1;

        BITensor _mul_split_output_1;
        // split 2 分支的代码
        BITensor _reshape_split_output_2;

        BITensor _transpose_split_output_2;

        BITensor _mul_split_output_2;
        // 矩乘输出
        BITensor _mat_mul_output;

        // 矩阵相加
        BITensor _mat_add_output;

        // Softmax的值进行输出
        BITensor _softmax_output;

        // MatMul进行运算
        BITensor _mat_mul_output_1;

        // 最后的Transpose
        BITensor _sum_transpose;

        // 最后的Reshape
        BITensor _reshape_sum_tensor;

        // 最后的Gemm出来的结果
        BITensor _gemm_sum_output;

        BITensor _final_reshape_output;

        // 复制的对象
        BINECopy _copy_f;
        // 是否已经完全初始化
        bool _is_prepared;
    };
}