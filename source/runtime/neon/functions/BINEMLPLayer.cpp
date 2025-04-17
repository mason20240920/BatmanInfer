//
// Created by Mason on 2025/4/7.
//

#include <runtime/neon/functions/BINEMLPLayer.hpp>

#include "common/utils/bi_log.hpp"
#include "data/core/utils/quantization/asymm_helpers.hpp"

namespace BatmanInfer {
    BINEMLPLayer::~BINEMLPLayer() = default;

    BINEMLPLayer::BINEMLPLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _rms_layer(),
        _quantization_layer(),
        _matrix_mul_core(),
        _c_proj(),
        _activation_layer(),
        _dequantization_layer(),
        _copy_f(),
        _memory_group(std::move(memory_manager)),
        _norm_output(),
        _norm_q_output(),
        _fc_q_output(),
        _proj_output(),
        _act_output(),
        _proj_input(),
        _gemm_lowp_output_stage(),
        _fc_s32_output(),
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

    void BINEMLPLayer::invert_qinfo_offset(BITensor &t) {
        BIQuantizationInfo qinfo = t.info()->quantization_info();
        t.info()->set_quantization_info(BIQuantizationInfo(qinfo.scale()[0], -qinfo.offset()[0], qinfo.is_dynamic()));
    }

    void BINEMLPLayer::dynamic_configure(const BIITensor *input, const size_t &seq_len, const size_t &batch_size) {
        _batch_size = batch_size;
        _seq_len = seq_len;
        _sub_norm_output_info.set_tensor_shape(BITensorShape(768, seq_len, batch_size));
        _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_output_info);
        _sub_norm_q_output_info.set_tensor_shape(BITensorShape(768, seq_len, batch_size));
        _sub_norm_q_output.allocator()->init(*_norm_q_output.allocator(), _sub_norm_q_output_info);
        _sub_fc_s32_output_info.set_tensor_shape(BITensorShape(3072, seq_len, batch_size));
        _sub_fc_s32_output.allocator()->init(*_fc_s32_output.allocator(), _sub_fc_s32_output_info);
        _sub_fc_q_output_info.set_tensor_shape(BITensorShape(3072, seq_len, batch_size));
        _sub_fc_q_output.allocator()->init(*_fc_q_output.allocator(), _sub_fc_q_output_info);
        _sub_act_output_info.set_tensor_shape(BITensorShape(3072, seq_len, batch_size));
        _sub_act_output.allocator()->init(*_act_output.allocator(), _sub_act_output_info);
        _sub_proj_input_info.set_tensor_shape(BITensorShape(3072, seq_len, batch_size));
        _sub_proj_input.allocator()->init(*_proj_input.allocator(), _sub_proj_input_info);
        _sub_proj_output_info.set_tensor_shape(BITensorShape(768, seq_len, batch_size));
        _sub_proj_output.allocator()->init(*_proj_output.allocator(), _sub_proj_output_info);

        _rms_layer.dynamic_configure(input);
        _quantization_layer.dynamic_configure(&_sub_norm_output);
        _matrix_mul_core.dynamic_configure(&_sub_norm_output, &_sub_fc_s32_output);
        _gemm_lowp_output_stage.dynamic_configure(&_sub_fc_s32_output);
        _activation_layer.dynamic_configure(&_sub_fc_q_output);
        _dequantization_layer.dynamic_configure(&_sub_act_output);
        _c_proj.dynamic_configure();
        _copy_f.dynamic_configure();
    }


    void BINEMLPLayer::configure(const BIITensor *input,
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
                                 // const BIActivationLayerInfo &act_info,
                                 BIITensor *output,
                                 const size_t &max_batch_size,
                                 const size_t &max_seq_len
    ) {
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
        _max_batch = max_batch_size;
        _max_seq = max_seq_len;

        // 中间变量输出的形状
        const auto norm_output_shape = BITensorShape(768, max_seq_len, max_batch_size); // 归一化输出
        const auto fc_fuse_output_shape = BITensorShape(3072, max_seq_len, max_batch_size); // Gemm + GeLU 融合操作

        // 初始化中间变量
        _norm_output.allocator()->init(BITensorInfo(norm_output_shape, 1, input->info()->data_type()));
        BIQuantizationInfo _norm_q_info{fc1_input_scale, fc1_input_zero_point};
        BIQuantizationInfo _fc_q_input_info{fc1_input_scale, -fc1_input_zero_point};
        BIQuantizationInfo _gelu_output_info{gelu_output_scale, gelu_output_zero_point};
        _norm_q_output.allocator()->init(BITensorInfo(norm_output_shape, 1, BIDataType::QASYMM8_SIGNED, _norm_q_info));
        BIQuantizationInfo _c_fc_q_info{fc1_output_scale, fc1_output_zero_point};
        _fc_q_output.allocator()->init(
            BITensorInfo(fc_fuse_output_shape, 1, BIDataType::QASYMM8_SIGNED, _c_fc_q_info));
        _fc_s32_output.allocator()->init(
            BITensorInfo(fc_fuse_output_shape, 1, BIDataType::S32));
        _act_output.allocator()->init(BITensorInfo(fc_fuse_output_shape, 1, BIDataType::QASYMM8_SIGNED,
                                                   _gelu_output_info));
        _proj_input.allocator()->init(BITensorInfo(fc_fuse_output_shape, 1, input->info()->data_type()));
        _proj_output.allocator()->init(BITensorInfo(norm_output_shape, 1, input->info()->data_type()));

        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_norm_q_output);
        _memory_group.manage(&_fc_q_output);
        _memory_group.manage(&_fc_s32_output);
        _memory_group.manage(&_act_output);
        _memory_group.manage(&_proj_input);
        _memory_group.manage(&_proj_output);

        _norm_output.allocator()->allocate();
        _norm_q_output.allocator()->allocate();
        _fc_q_output.allocator()->allocate();
        _fc_s32_output.allocator()->allocate();
        _act_output.allocator()->allocate();
        _proj_input.allocator()->allocate();
        _proj_output.allocator()->allocate();

        const auto sub_norm_output_shape = BITensorShape(768, _seq_len, _batch_size);
        _sub_norm_output_info = BITensorInfo(sub_norm_output_shape, 1, input->info()->data_type());
        _sub_norm_output_info.set_format(Format::F16);
        _sub_norm_output.allocator()->init(_sub_norm_output_info);

        _sub_norm_q_output_info = BITensorInfo(sub_norm_output_shape, 1, BIDataType::QASYMM8_SIGNED, _norm_q_info);
        _sub_norm_q_output_info.set_format(Format::S8);
        _sub_norm_q_output.allocator()->init(_sub_norm_q_output_info);

        const auto sub_fc_fuse_output_shape = BITensorShape(3072, _seq_len, _batch_size);
        _sub_fc_s32_output_info = BITensorInfo(sub_fc_fuse_output_shape, 1, BIDataType::S32);
        _sub_fc_s32_output_info.set_format(Format::S32);
        _sub_fc_s32_output.allocator()->init(_sub_fc_s32_output_info);

        _sub_fc_q_output_info = BITensorInfo(sub_fc_fuse_output_shape, 1, BIDataType::QASYMM8_SIGNED, _c_fc_q_info);
        _sub_fc_q_output_info.set_format(Format::S8);
        _sub_fc_q_output.allocator()->init(_sub_fc_q_output_info);

        _sub_act_output_info = BITensorInfo(sub_fc_fuse_output_shape, 1, BIDataType::QASYMM8_SIGNED, _gelu_output_info);
        _sub_act_output_info.set_format(Format::S8);
        _sub_act_output.allocator()->init(_sub_act_output_info);

        _sub_proj_input_info = BITensorInfo(sub_fc_fuse_output_shape, 1, input->info()->data_type());
        _sub_proj_input_info.set_format(Format::F16);
        _sub_proj_input.allocator()->init(_sub_proj_input_info);

        _sub_proj_output_info = BITensorInfo(sub_norm_output_shape, 1, input->info()->data_type());
        _sub_proj_output_info.set_format(Format::F16);
        _sub_proj_output.allocator()->init(_sub_proj_output_info);


        _rms_layer.configure(input, gamma, &_sub_norm_output);
        _quantization_layer.configure(&_sub_norm_output, &_sub_norm_q_output);
        invert_qinfo_offset(_sub_norm_q_output);
        GEMMInfo gemm_info = GEMMInfo(false,
                                      false,
                                      true,
                                      false,
                                      false,
                                      false,
                                      BIGEMMLowpOutputStageInfo(),
                                      false, true, false,
                                      BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
        _matrix_mul_core.configure(&_sub_norm_q_output, fc_weights, nullptr, &_sub_fc_s32_output, gemm_info);
        BIGEMMLowpOutputStageInfo c_fc_stage_info;
        c_fc_stage_info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        c_fc_stage_info.output_data_type = BIDataType::QASYMM8_SIGNED; // 设置输c出数据类型
        c_fc_stage_info.is_quantized_per_channel = true; // 因为权重是per-channel量化// 设置输出范围
        c_fc_stage_info.gemmlowp_offset = fc1_output_zero_point;
        c_fc_stage_info.gemmlowp_min_bound = -128; // 通常是-128
        c_fc_stage_info.gemmlowp_max_bound = 127; // 通常是127// 假设已有输入tensor的量化参数
        // 使用calculate_quantized_multipliers计算每个通道的multiplier和shift
        // 这个函数会自动填充gemmlowp_multipliers和gemmlowp_shifts
        quantization::calculate_quantized_multipliers(_fc_q_input_info,
                                                      *c_fc_weight_qinfo,
                                                      _c_fc_q_info,
                                                      c_fc_stage_info);
        _gemm_lowp_output_stage.configure(&_sub_fc_s32_output, fc_bias, &_sub_fc_q_output, c_fc_stage_info);
        _activation_layer.configure(&_sub_fc_q_output, &_sub_act_output,
                                    BIActivationLayerInfo(BIActivationFunction::GELU));
        _dequantization_layer.configure(&_sub_act_output, &_sub_proj_input);
        _c_proj.configure(&_sub_proj_input, proj_weights, proj_bias, &_sub_proj_output, 1.0f, 1.0f, gemm_info);
        _copy_f.configure(&_sub_proj_output, output);
    }

    void BINEMLPLayer::run() {
        prepare(); // 内存分配管理
        _rms_layer.run();
        // invert_qinfo_offset(_norm_q_output);
        _quantization_layer.run();
        // invert_qinfo_offset(_norm_q_output);
        _matrix_mul_core.run();
        _gemm_lowp_output_stage.run();
        _activation_layer.run();
        _dequantization_layer.run();
        _c_proj.run();
        _copy_f.run();
    }

    void BINEMLPLayer::prepare() {
        if (!_is_prepared) {
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_output_info);
            _sub_norm_q_output.allocator()->init(*_norm_q_output.allocator(), _sub_norm_q_output_info);
            _sub_fc_s32_output.allocator()->init(*_fc_s32_output.allocator(), _sub_fc_s32_output_info);
            _sub_fc_q_output.allocator()->init(*_fc_q_output.allocator(), _sub_fc_q_output_info);
            _sub_act_output.allocator()->init(*_act_output.allocator(), _sub_act_output_info);
            _sub_proj_input.allocator()->init(*_proj_input.allocator(), _sub_proj_input_info);
            _sub_proj_output.allocator()->init(*_proj_output.allocator(), _sub_proj_output_info);
            _is_prepared = true;
        }
    }
}
