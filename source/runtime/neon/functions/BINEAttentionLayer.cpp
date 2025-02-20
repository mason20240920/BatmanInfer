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

    BINEAttentionLayer::BINEAttentionLayer(std::shared_ptr<BIIMemoryManager> memory_manager) :
            _memory_group(std::move(memory_manager)),
            _normalization_layer(),
            _gemm_state_f(),
            _split_layer(),
            _reshape_1_f(),
            _transpose_1_f(),
            _reshape_2_f(),
            _transpose_2_f(),
            _reshape_3_f(),
            _transpose_3_f(),
            _mul_1_f(),
            _mul_2_f(),
            _gemm_fuse_f(),
            _matmul_2_f(),
            _softmax_layer(),
            _transpose_final_f(),
            _reshape_final_f(),
            _gemm_final_f(),
            _norm_output(),
            _gemm_output(),
            _split_result_0(),
            _split_result_1(),
            _split_result_2(),
            _reshape_1_output(),
            _transpose_1_output(),
            _reshape_2_output(),
            _transpose_2_output(),
            _reshape_3_output(),
            _transpose_3_output(),
            _mul_1_output(),
            _mul_2_output(),
            _gemm_fuse_output(),
            _softmax_output(),
            _matmul_2_output(),
            _transpose_final_output(),
            _reshape_final_output(),
            _gemm_final_output(),
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

    void BINEAttentionLayer::configure(const BatmanInfer::BIITensor *input,
                                       const BatmanInfer::BIITensor *weights,
                                       const BatmanInfer::BIITensor *bias,
                                       const BIITensor *scalar,
                                       const BIITensor *add_weights,
                                       const BIITensor *weights_second,
                                       const BIITensor *bias_second,
                                       const BIITensor *gamma,
                                       const PermutationVector &perm,
                                       const PermutationVector &perm2,
                                       const PermutationVector &final_perm,
                                       const size_t &hidden_size,
                                       const size_t &max_seq_len,
                                       const size_t &batch_size,
                                       BatmanInfer::BIITensor *output) {
        // 结果判断
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, gamma, output);  // 输入的参数是否为空
        BI_COMPUTE_ERROR_THROW_ON(BINEAttentionLayer::validate(input->info(), weights->info(),
                                                               bias->info(), output->info())); // 验证输入, 权重，偏置和输出信息
        BI_COMPUTE_LOG_PARAMS(input, weights, bias, output); // 获取log的参数

        // 配置私有参数
        _max_seq_len = max_seq_len; // 最大的值
        _hidden_size = hidden_size; // 隐藏层长度
        _batch_size = batch_size; // 最大块
        _is_prepared = false;  // 初始化标志，标识尚未准备好

        // 配置中间张量输出
        BITensorShape normal_shape = BITensorShape(_hidden_size, _max_seq_len, _batch_size); // 默认输入和输出
        BITensorShape gemm_1_shape = BITensorShape(weights->info()->tensor_shape()[0], _max_seq_len,
                                                   _batch_size); // 第一个gemm的输出
        BITensorShape split_shape = BITensorShape(_hidden_size, _max_seq_len, _batch_size);
        BITensorShape reshape_split_shape = BITensorShape(64, 12, _max_seq_len, _batch_size);
        BITensorShape transpose_1_shape = BITensorShape(64, _max_seq_len, 12, _batch_size);
        BITensorShape transpose_2_shape = BITensorShape(_max_seq_len, 64, 12, _batch_size);
        BITensorShape matmul_1_shape = BITensorShape(_max_seq_len, _max_seq_len, 12, _batch_size);

        _norm_output.allocator()->init(BITensorInfo(normal_shape, 1, input->info()->data_type()));
        _gemm_output.allocator()->init(BITensorInfo(gemm_1_shape, 1, input->info()->data_type()));
        _split_result_0.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));
        _split_result_1.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));
        _split_result_2.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));
        _reshape_1_output.allocator()->init(BITensorInfo(reshape_split_shape, 1, input->info()->data_type()));
        _transpose_1_output.allocator()->init(BITensorInfo(transpose_1_shape, 1, input->info()->data_type()));
        _reshape_2_output.allocator()->init(BITensorInfo(reshape_split_shape, 1, input->info()->data_type()));
        _transpose_2_output.allocator()->init(BITensorInfo(transpose_2_shape, 1, input->info()->data_type()));
        _reshape_3_output.allocator()->init(BITensorInfo(reshape_split_shape, 1, input->info()->data_type()));
        _transpose_3_output.allocator()->init(BITensorInfo(transpose_1_shape, 1, input->info()->data_type()));
        _mul_1_output.allocator()->init(BITensorInfo(transpose_2_shape, 1, input->info()->data_type()));
        _mul_2_output.allocator()->init(BITensorInfo(transpose_1_shape, 1, input->info()->data_type()));
        _gemm_fuse_output.allocator()->init(BITensorInfo(matmul_1_shape, 1, input->info()->data_type()));
        _softmax_output.allocator()->init(BITensorInfo(matmul_1_shape, 1, input->info()->data_type()));
        _matmul_2_output.allocator()->init(BITensorInfo(transpose_1_shape, 1, input->info()->data_type()));
        _transpose_final_output.allocator()->init(BITensorInfo(reshape_split_shape, 1, input->info()->data_type()));
        _reshape_final_output.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));
        _gemm_final_output.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));


        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_gemm_output);
        _memory_group.manage(&_split_result_0);
        _memory_group.manage(&_split_result_1);
        _memory_group.manage(&_split_result_2);
        _memory_group.manage(&_reshape_1_output);
        _memory_group.manage(&_transpose_1_output);
        _memory_group.manage(&_reshape_2_output);
        _memory_group.manage(&_transpose_2_output);
        _memory_group.manage(&_reshape_3_output);
        _memory_group.manage(&_transpose_3_output);
        _memory_group.manage(&_mul_1_output);
        _memory_group.manage(&_mul_2_output);
        _memory_group.manage(&_gemm_fuse_output);
        _memory_group.manage(&_matmul_2_output);
        _memory_group.manage(&_softmax_output);
        _memory_group.manage(&_transpose_final_output);
        _memory_group.manage(&_reshape_final_output);
        _memory_group.manage(&_gemm_final_output);

        _norm_output.allocator()->allocate();
        _gemm_output.allocator()->allocate();
        _split_result_0.allocator()->allocate();
        _split_result_1.allocator()->allocate();
        _split_result_2.allocator()->allocate();
        _reshape_1_output.allocator()->allocate();
        _transpose_1_output.allocator()->allocate();
        _reshape_2_output.allocator()->allocate();
        _transpose_2_output.allocator()->allocate();
        _reshape_3_output.allocator()->allocate();
        _transpose_3_output.allocator()->allocate();
        _mul_1_output.allocator()->allocate();
        _mul_2_output.allocator()->allocate();
        _gemm_fuse_output.allocator()->allocate();
        _softmax_output.allocator()->allocate();
        _matmul_2_output.allocator()->allocate();
        _transpose_final_output.allocator()->allocate();
        _reshape_final_output.allocator()->allocate();
        _gemm_final_output.allocator()->allocate();

        // 配置层的效果
        _normalization_layer.configure(input, gamma, &_norm_output);
        GEMMInfo gemm_info;
        gemm_info.set_fast_math(true);
        _gemm_state_f.configure(&_norm_output, weights, bias, &_gemm_output, 1.0f, 1.0f, gemm_info);
        std::vector<BIITensor *> outputs = {&_split_result_1, &_split_result_2, &_split_result_0};
        _split_layer.configure(&_gemm_output, outputs, 0);
        _reshape_1_f.configure(&_split_result_0, &_reshape_1_output);
        _transpose_1_f.configure(&_reshape_1_output, &_transpose_1_output, perm);
        _reshape_2_f.configure(&_split_result_1, &_reshape_2_output);
        _transpose_2_f.configure(&_reshape_2_output, &_transpose_2_output, perm2);
        _mul_1_f.configure(&_transpose_2_output,
                           scalar,
                           &_mul_1_output,
                           1.0f,
                           BIConvertPolicy::WRAP,
                           BIRoundingPolicy::TO_ZERO);
        _reshape_3_f.configure(&_split_result_2, &_reshape_3_output);
        _transpose_3_f.configure(&_reshape_3_output, &_transpose_3_output, perm);
        _mul_2_f.configure(&_transpose_3_output,
                           scalar,
                           &_mul_2_output,
                           1.0f,
                           BIConvertPolicy::WRAP,
                           BIRoundingPolicy::TO_ZERO);

        // Define MatMulInfo
        BIMatMulInfo matmul_info; // No transpose for lhs or rhs

        // Define CpuMatMulSettings
        BICpuMatMulSettings settings;
        // Enable fast math for optimization
        settings = settings.fast_math(true);
        // 设置不是常量
        _mul_1_output.info()->set_are_values_constant(false);
        _mul_2_output.info()->set_are_values_constant(false);
        _gemm_fuse_f.configure(&_mul_2_output, &_mul_1_output, add_weights, &_gemm_fuse_output, 1.0f, 1.0f, gemm_info);
        _softmax_layer.configure(&_gemm_fuse_output,
                                 &_softmax_output,
                                 1.0f,
                                 0);
        _transpose_1_output.info()->set_are_values_constant(false);
        _softmax_output.info()->set_are_values_constant(false);
        _matmul_2_f.configure(&_softmax_output,
                              &_transpose_1_output,
                              &_matmul_2_output, matmul_info, settings);
        _transpose_final_f.configure(&_matmul_2_output, &_transpose_final_output, final_perm);
        _reshape_final_f.configure(&_transpose_final_output, &_reshape_final_output);
        _gemm_final_f.configure(&_reshape_final_output, weights_second, bias_second, &_gemm_final_output, 1.f, 1.f,
                                gemm_info);
        _copy_f.configure(&_gemm_final_output, output);
    }

    void BINEAttentionLayer::run() {
//        prepare();

        BIMemoryGroupResourceScope scope_mg(_memory_group);

        // 执行函数
        _normalization_layer.run(); // 归一化 layer norm
        _gemm_state_f.run();
        _split_layer.run();
        _reshape_1_f.run();
        _transpose_1_f.run();
        _reshape_2_f.run();
        _transpose_2_f.run();
        _mul_1_f.run();
        _reshape_3_f.run();
        _transpose_3_f.run();
        _mul_2_f.run();
        _gemm_fuse_f.run();
        _softmax_layer.run();
        _matmul_2_f.run();
        _transpose_final_f.run();
        _reshape_final_f.run();
        _gemm_final_f.run();
        _copy_f.run(); // 运行拷贝

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