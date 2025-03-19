//
// Created by Mason on 2025/2/9.
//

#include <runtime/neon/functions/BINEAttentionLowpLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <function_info/bi_MatMulInfo.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

namespace BatmanInfer {
    BINEAttentionLowpLayer::~BINEAttentionLowpLayer() = default;

    BINEAttentionLowpLayer::BINEAttentionLowpLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)),
        _rms_norm_layer(),
        _quantization_layer(),
        _dequantization_layer(),
        _copy_f(),
        _norm_output(),
        _quantization_output(),
        _dequantization_output(),
        _is_prepared(false) {
    }

    BIStatus
    BINEAttentionLowpLayer::validate(const BatmanInfer::BIITensorInfo *input,
                                     const BatmanInfer::BIITensorInfo *weights,
                                     const BatmanInfer::BIITensorInfo *bias,
                                     const BatmanInfer::BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F32, BIDataType::F16);

        BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 3);

        return BIStatus{};
    }

    void BINEAttentionLowpLayer::configure(const BatmanInfer::BIITensor *input,
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
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, bias, gamma, output); // 输入的参数是否为空
        BI_COMPUTE_ERROR_THROW_ON(BINEAttentionLayer::validate(input->info(), weights->info(),
            bias->info(), output->info())); // 验证输入, 权重，偏置和输出信息
        BI_COMPUTE_LOG_PARAMS(input, weights, bias, output); // 获取log的参数

        // 配置私有参数
        _max_seq_len = max_seq_len; // 最大的值
        _hidden_size = hidden_size; // 隐藏层长度
        _batch_size = batch_size; // 最大块
        _is_prepared = false; // 初始化标志，标识尚未准备好

        // 配置中间张量输出
        BITensorShape normal_shape = BITensorShape(_hidden_size, _max_seq_len, _batch_size); // 默认输入和输出
        BITensorShape gemm_1_shape = BITensorShape(weights->info()->tensor_shape()[0], _max_seq_len,
                                                   _batch_size); // 第一个gemm的输出


        _norm_output.allocator()->init(BITensorInfo(normal_shape, 1, input->info()->data_type()));
        _quantization_output.allocator()->
                init(BITensorInfo(normal_shape, 1, BIDataType::QASYMM8_SIGNED, BIQuantizationInfo(0.5)));
        _dequantization_output.allocator()->
                init(BITensorInfo(normal_shape, 1, input->info()->data_type()));
        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_quantization_output);
        _memory_group.manage(&_dequantization_output);

        _norm_output.allocator()->allocate();
        _quantization_output.allocator()->allocate();
        _dequantization_output.allocator()->allocate();


        // 配置量化信息
        _rms_norm_layer.configure(input, gamma, &_norm_output);
        _quantization_layer.configure(&_norm_output, &_quantization_output);
        _dequantization_layer.configure(&_quantization_output, &_dequantization_output);
        _copy_f.configure(&_dequantization_output, output);
    }

    void BINEAttentionLowpLayer::run() {
        prepare();

        BIMemoryGroupResourceScope scope_mg(_memory_group);

        // 运行计算
        _rms_norm_layer.run(); // 归一化计算
        _quantization_layer.run(); // 量化计算
        _dequantization_layer.run(); // 反量化计算

        // 拷贝隐藏层到输出
        _copy_f.run();
    }

    void BINEAttentionLowpLayer::prepare() {
        if (!_is_prepared) {
            //            _reshape.prepare();
            //            _gemm_state_f.prepare();

            _is_prepared = true;
        }
    }
}
