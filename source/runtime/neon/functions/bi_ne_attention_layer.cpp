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


        // 将_reshape_output和_gemm_output交给内存管理器管理
        _memory_group.manage(&_reshape_output);

        _reshape.configure(input, &_reshape_output);

        _memory_group.manage(&_gemm_output);


        _gemm_state_f.configure(weights, &_reshape_output, bias, &_gemm_output, 1.f, 1.f);

        _memory_group.manage(&_reshape_output_2);

        _reshape2.configure(&_gemm_output, &_reshape_output_2);

        // 内存管理
        _memory_group.manage(&_split_result_0);
        _memory_group.manage(&_split_result_1);
        _memory_group.manage(&_split_result_2);

        _reshape_output.allocator()->allocate();
        _gemm_output.allocator()->allocate();
        _reshape_output_2.allocator()->allocate();
        _split_result_0.allocator()->allocate();
        _split_result_1.allocator()->allocate();
        _split_result_2.allocator()->allocate();

        // 进行切分层的结果输入和输出
        std::vector<BIITensor *> outputs = {&_split_result_0, &_split_result_1, &_split_result_2};


        _split_layer.configure(&_reshape_output_2, outputs, 2);

        _copy_f.configure(&_split_result_0, output);

    }

    void BINEAttentionLayer::run() {
//        prepare();

        BIMemoryGroupResourceScope scope_mg(_memory_group);

        _reshape.run();

//        _reshape_output.print(_reshape_output)

        _gemm_state_f.run();

        _reshape2.run();

        _split_layer.run();

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