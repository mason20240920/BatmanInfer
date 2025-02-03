//
// Created by Mason on 2025/1/23.
//

#include <runtime/neon/functions/bi_ne_attention_layer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

namespace BatmanInfer {
    BINEAttentionLayer::~BINEAttentionLayer() = default;

    BINEAttentionLayer::BINEAttentionLayer(std::shared_ptr<BIIMemoryManager> memory_manager) :
            _memory_group(std::move(memory_manager)),
            _gemm_state_f(),
            _reshape(),
            _reshape2(),
            _gemm_output(),
            _reshape_output(),
            _reshape_output_2(),
            _split_result_0(),
            _split_result_1(),
            _split_result_2(),
            _split_layer(),
            _reshape_split_0(),
            _reshape_split_output_0(),
            _transpose_split_0(),
            _transpose_split_output_0(),
            _reshape_split_1(),
            _transpose_split_1(),
            _reshape_split_2(),
            _transpose_split_2(),
            _mul_op_0(),
            _mul_op_1(),
            _reshape_split_output_1(),
            _transpose_split_output_1(),
            _mul_split_output_1(),
            _reshape_split_output_2(),
            _transpose_split_output_2(),
            _mul_split_output_2(),
            _matmul_op(),
            _mat_mul_output(),
            _is_prepared(false) {

    }

    BIStatus
    BINEAttentionLayer::validate(const BatmanInfer::BIITensorInfo *input,
                                 const BatmanInfer::BIITensorInfo *weights,
                                 const BatmanInfer::BIITensorInfo *bias,
                                 const BatmanInfer::BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F16, BIDataType::F32);

        BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 3);

        return BIStatus{};
    }

    void BINEAttentionLayer::configure(const BatmanInfer::BIITensor *input,
                                       const BatmanInfer::BIITensor *weights,
                                       const BatmanInfer::BIITensor *bias,
                                       const BIITensor *scalar,
                                       const PermutationVector &perm,
                                       const PermutationVector &perm2,
                                       BatmanInfer::BIITensor *output) {
        // 输入的参数是否为空
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, output);
        // 验证输入, 权重，偏置和输出信息
        BI_COMPUTE_ERROR_THROW_ON(BINEAttentionLayer::validate(input->info(), weights->info(),
                                                               bias->info(), output->info()));
        // 获取log的参数
        BI_COMPUTE_LOG_PARAMS(input, weights, bias, output);

        // 转置的输出shape
        BITensorShape reshape_shape = BITensorShape(16, 768);

        // Gmm的输出reshape
        BITensorShape gemm_shape = BITensorShape(16, 2304);

        // 第二层的Reshape
        BITensorShape reshape_shape_2 = BITensorShape(1, 16, 2304);

        // 第三进行结果的Split
        BITensorShape split_shape = BITensorShape(1, 16, 768);

        // 第一层切分的reshape结构
        BITensorShape split_layer_0_shape = BITensorShape(1, 16, 12, 64);

        // 第一层切分的transpose结构
        BITensorShape split_layer_0_trans_shape = BITensorShape(64, 16, 12, 1);

        // 第二个分支的split的transpose不同
        BITensorShape split_linear_1_trans_shape = BITensorShape(16, 64, 12, 1);

        // 合并分支代码
        BITensorShape mat_mul_shape = BITensorShape(16, 16, 12, 1);

        // 初始化标志，标识尚未准备好
        _is_prepared = false;

        // 初始化中间张量 _reshape_output, 用于存储Reshape的输出
        _reshape_output.allocator()->init(BITensorInfo(reshape_shape, 1, input->info()->data_type()));

        _gemm_output.allocator()->init(BITensorInfo(gemm_shape, 1, input->info()->data_type()));

        _reshape_output_2.allocator()->init(BITensorInfo(reshape_shape_2, 1, input->info()->data_type()));

        // 结果进行切分
        _split_result_0.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));

        _split_result_1.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));

        _split_result_2.allocator()->init(BITensorInfo(split_shape, 1, input->info()->data_type()));

        // 初始化split_0的推理分支
        _reshape_split_output_0.allocator()->init(BITensorInfo(split_layer_0_shape, 1, input->info()->data_type()));

        _transpose_split_output_0.allocator()->init(
                BITensorInfo(split_layer_0_trans_shape, 1, input->info()->data_type()));

        // 初始化split_1的推理分支
        _reshape_split_output_1.allocator()->init(BITensorInfo(split_layer_0_shape, 1, input->info()->data_type()));
        _transpose_split_output_1.allocator()->init(
                BITensorInfo(split_linear_1_trans_shape, 1, input->info()->data_type()));
        _mul_split_output_1.allocator()->init(BITensorInfo(split_linear_1_trans_shape, 1, input->info()->data_type()));

        // 初始化split_2的推理分支
        _reshape_split_output_2.allocator()->init(BITensorInfo(split_layer_0_shape, 1, input->info()->data_type()));
        _transpose_split_output_2.allocator()->init(
                BITensorInfo(split_layer_0_trans_shape, 1, input->info()->data_type()));
        _mul_split_output_2.allocator()->init(BITensorInfo(split_layer_0_trans_shape, 1, input->info()->data_type()));

        // 推理分支代码合并
        _mat_mul_output.allocator()->init(BITensorInfo(mat_mul_shape, 1, input->info()->data_type()));


        // 内存管理
        _memory_group.manage(&_reshape_split_output_0);
        _memory_group.manage(&_transpose_split_output_0);
        _memory_group.manage(&_reshape_split_output_1);
        _memory_group.manage(&_transpose_split_output_1);
        _memory_group.manage(&_mul_split_output_1);
        _memory_group.manage(&_reshape_split_output_2);
        _memory_group.manage(&_transpose_split_output_2);
        _memory_group.manage(&_mul_split_output_2);
        _memory_group.manage(&_gemm_output);
        _memory_group.manage(&_reshape_output_2);
        _memory_group.manage(&_mat_mul_output);
        // 将_reshape_output和_gemm_output交给内存管理器管理
        _memory_group.manage(&_reshape_output);
        _memory_group.manage(&_split_result_0);
        _memory_group.manage(&_split_result_1);
        _memory_group.manage(&_split_result_2);

        _reshape_output.allocator()->allocate();
        _gemm_output.allocator()->allocate();
        _reshape_output_2.allocator()->allocate();
        _split_result_0.allocator()->allocate();
        _split_result_1.allocator()->allocate();
        _split_result_2.allocator()->allocate();
        _reshape_split_output_0.allocator()->allocate();
        _transpose_split_output_0.allocator()->allocate();
        _reshape_split_output_1.allocator()->allocate();
        _transpose_split_output_1.allocator()->allocate();
        _mul_split_output_1.allocator()->allocate();
        _reshape_split_output_2.allocator()->allocate();
        _transpose_split_output_2.allocator()->allocate();
        _mul_split_output_2.allocator()->allocate();
        _mat_mul_output.allocator()->allocate();



        // 进行切分层的结果输入和输出
        _reshape.configure(input, &_reshape_output);
        _gemm_state_f.configure(weights, &_reshape_output, bias, &_gemm_output, 1.f, 1.f);
        _reshape2.configure(&_gemm_output, &_reshape_output_2);
        std::vector<BIITensor *> outputs = {&_split_result_0, &_split_result_1, &_split_result_2};
        _split_layer.configure(&_reshape_output_2, outputs, 2);
        _reshape_split_0.configure(&_split_result_0, &_reshape_split_output_0);
        _transpose_split_0.configure(&_reshape_split_output_0, &_transpose_split_output_0, perm);
        _reshape_split_1.configure(&_split_result_1, &_reshape_split_output_1);
        _transpose_split_1.configure(&_reshape_split_output_1, &_transpose_split_output_1, perm2);
        _mul_op_0.configure(&_transpose_split_output_1,
                            scalar,
                            &_mul_split_output_1,
                            1.0f,
                            BIConvertPolicy::WRAP,
                            BIRoundingPolicy::TO_ZERO);
        _reshape_split_2.configure(&_split_result_2, &_reshape_split_output_2);
        _transpose_split_2.configure(&_reshape_split_output_2, &_transpose_split_output_2, perm);
        _mul_op_1.configure(&_transpose_split_output_2,
                            scalar,
                            &_mul_split_output_2,
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
        _mul_split_output_1.info()->set_are_values_constant(false);
        _mul_split_output_2.info()->set_are_values_constant(false);
        _matmul_op.configure(&_mul_split_output_1,
                             &_mul_split_output_2,
                             &_mat_mul_output, matmul_info, settings);

        _copy_f.configure(&_mat_mul_output, output);

    }

    void BINEAttentionLayer::run() {
//        prepare();

        BIMemoryGroupResourceScope scope_mg(_memory_group);

        _reshape.run();

//        _reshape_output.print(_reshape_output)

        _gemm_state_f.run();

        _reshape2.run();

        _split_layer.run();

        // 进行第一个Split推理分支进行切分
        _reshape_split_0.run();
        _transpose_split_0.run();

        // 进行第二个Split推理分支进行运行
        _reshape_split_1.run();
        _transpose_split_1.run();
        _mul_op_0.run();

        // 进行第三个Split推理分支进行运行
        _reshape_split_2.run();
        _transpose_split_2.run();
        _mul_op_1.run();

        // 进行合并矩阵计算
        _matmul_op.run();

        // 拷贝隐藏层到输出
        _copy_f.run();
    }

    void BINEAttentionLayer::prepare() {
        if (!_is_prepared) {
//            _reshape.prepare();
//            _gemm_state_f.prepare();

            _is_prepared = true;
        }
    }
}