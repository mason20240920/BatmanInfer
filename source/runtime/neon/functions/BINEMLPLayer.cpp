//
// Created by Mason on 2025/4/7.
//

#include <runtime/neon/functions/BINEMLPLayer.hpp>

#include "common/utils/bi_log.hpp"

namespace BatmanInfer {
    BINEMLPLayer::~BINEMLPLayer() = default;

    BINEMLPLayer::BINEMLPLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _rms_layer(),
        // _quantization_layer(),
        // _matrix_mul_core(),
        // _c_proj(),
        // _dequantization_layer(), _copy_f(),
        _memory_group(std::move(memory_manager)),
        _norm_output(),
        _norm_q_output(),
        // _fuse_output(),
        // _proj_output(),
        // _proj_q_output(),
        _max_batch(0), _max_seq(0) {
    }

    BIStatus BINEMLPLayer::validate(const BIITensorInfo *input,
                                    const BIITensorInfo *fc_weights,
                                    const BIITensorInfo *fc_bias,
                                    const BIITensorInfo *proj_weights,
                                    const BIITensorInfo *proj_bias,
                                    const BIITensorInfo *gamma,
                                    const BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(
            input, fc_weights, fc_bias, proj_weights, proj_bias, output, gamma);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F16);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(fc_weights, BIDataType::QSYMM8_PER_CHANNEL);

        BI_COMPUTE_RETURN_ERROR_ON(input->tensor_shape() != output->tensor_shape());

        return BIStatus{};
    }


    void BINEMLPLayer::configure(const BIITensor *input,
                                 const BIITensor *fc_weights,
                                 const BIITensor *fc_bias,
                                 const BIITensor *proj_weights,
                                 const BIITensor *proj_bias,
                                 const BIITensor *gamma,
                                 const BIActivationLayerInfo &act_info,
                                 BIITensor *output,
                                 const size_t &batch_size,
                                 const size_t &seq_len) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, fc_weights, fc_bias, proj_weights, proj_bias, output, gamma);
        BI_COMPUTE_ERROR_THROW_ON(
            BINEMLPLayer::validate(input->info(),
                fc_weights->info(),
                fc_bias->info(),
                proj_weights->info(),
                proj_bias->info(),
                gamma->info(),
                output->info()));

        BI_COMPUTE_LOG_PARAMS(input, output);
        _max_batch = batch_size;
        _max_seq = seq_len;

        // 中间变量输出的形状
        BITensorShape norm_output_shape = BITensorShape(input->info()->tensor_shape()); // 归一化输出
        BITensorShape fc_fuse_output_shape = BITensorShape(3072, seq_len, batch_size); // Gemm + GeLU 融合操作
        BITensorShape proj_output_shape = BITensorShape(output->info()->tensor_shape()); // 最后降解的操作

        // 初始化中间变量
        _norm_output.allocator()->init(BITensorInfo(norm_output_shape, 1, input->info()->data_type()));
        _norm_q_output.allocator()->init(BITensorInfo(norm_output_shape, 1, input->info()->data_type()));
        // _fuse_output.allocator()->init(BITensorInfo(fc_fuse_output_shape, 1, input->info()->data_type()));
        // _proj_output.allocator()->init(BITensorInfo(proj_output_shape, 1, input->info()->data_type()));
        // _proj_q_output.allocator()->init(BITensorInfo(proj_output_shape, 1, input->info()->data_type()));

        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_norm_q_output);
        // _memory_group.manage(&_fuse_output);
        // _memory_group.manage(&_proj_output);
        // _memory_group.manage(&_proj_q_output);

        _norm_output.allocator()->allocate();
        _norm_q_output.allocator()->allocate();
        // _fuse_output.allocator()->allocate();
        // _proj_output.allocator()->allocate();
        // _proj_q_output.allocator()->allocate();

        // 配置Gemm操作
        GEMMInfo fc_gemm_info, proj_gemm_info;
        fc_gemm_info.set_activation_info(act_info);
        fc_gemm_info.set_fast_math(true);
        proj_gemm_info.set_fast_math(true);

        _rms_layer.configure(input, gamma, &_norm_output);
        // _quantization_layer.configure(&_norm_output, &_norm_q_output);
        // _matrix_mul_core.configure(&_norm_q_output, fc_weights, fc_bias, &_fuse_output, fc_gemm_info);
        // _c_proj.configure(&_fuse_output, proj_weights, proj_bias, &_proj_q_output, proj_gemm_info);
        // _dequantization_layer.configure(&_proj_q_output, &_proj_output);
        // _copy_f.configure(&_proj_output, output);
    }

    void BINEMLPLayer::run() {
        BIMemoryGroupResourceScope scope_mg(_memory_group);
        _rms_layer.run();
        // _quantization_layer.run();
        // _matrix_mul_core.run();
        // _c_proj.run();
        // _dequantization_layer.run();
        // _copy_f.run();
    }
}
