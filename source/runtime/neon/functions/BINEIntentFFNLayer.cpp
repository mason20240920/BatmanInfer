//
// Created by Mason on 2025/8/6.
//

#include <runtime/neon/functions/BINEIntentFFNLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

namespace BatmanInfer {
    BINEIntentFFNLayer::~BINEIntentFFNLayer() = default;

    BINEIntentFFNLayer::BINEIntentFFNLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)),
        _layer_norm_layer(),
        _c_fc_fuse_act(),
        _c_proj(),
        _copy_f(),
        _norm_output(),
        _fuse_output(),
        _proj_output(),
        _max_batch(0),
        _max_seq(0),
        _is_prepared(false) {
    }

    void  BINEIntentFFNLayer::dynamic_configure(const BIITensor *input,
                                                const size_t &batch_size,
                                                const size_t &seq_len) {
        _batch_size = batch_size;
        _seq_len = seq_len;
        _sub_norm_output_info.set_tensor_shape(BITensorShape(768, _seq_len, batch_size));
        _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_output_info);
        _sub_fuse_output_info.set_tensor_shape(BITensorShape(3072, _seq_len, batch_size));
        _sub_fuse_output.allocator()->init(*_fuse_output.allocator(), _sub_fuse_output_info);
        _sub_proj_output_info.set_tensor_shape(BITensorShape(768, _seq_len, batch_size));
        _sub_proj_output.allocator()->init(*_proj_output.allocator(), _sub_proj_output_info);

        _layer_norm_layer.dynamic_configure(input);
        _c_fc_fuse_act.dynamic_configure();
        _c_proj.dynamic_configure();
        _copy_f.dynamic_configure();
    }

    BIStatus
     BINEIntentFFNLayer::validate(const BIITensorInfo *input,
                                   const BIITensorInfo *fc_weights,
                                   const BIITensorInfo *fc_bias,
                                   const BIITensorInfo *proj_weights,
                                   const BIITensorInfo *proj_bias,
                                   const BIITensorInfo *gamma,
                                   const BIITensorInfo *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, fc_weights, fc_bias, proj_weights, proj_bias, output, gamma);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F16, BIDataType::F32);

        BI_COMPUTE_RETURN_ERROR_ON(input->tensor_shape() != output->tensor_shape());

        return BIStatus{};
    }

    void BINEIntentFFNLayer::configure(const BatmanInfer::BIITensor *input,
                                         const BatmanInfer::BIITensor *fc_weights,
                                         const BatmanInfer::BIITensor *fc_bias,
                                         const BatmanInfer::BIITensor *proj_weights,
                                         const BatmanInfer::BIITensor *proj_bias,
                                         const BatmanInfer::BIITensor *gamma,
                                         const BIITensor *ln_2_bias,
                                         const BatmanInfer::BIActivationLayerInfo &act_info,
                                         BatmanInfer::BIITensor *output,
                                         const size_t &max_batch_size,
                                         const size_t &max_seq_len,
                                         const size_t &cur_batch_size,
                                         const size_t &cur_seq_size) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, fc_weights, fc_bias, proj_weights, proj_bias, output); // 输入的参数是否为空

        BI_COMPUTE_ERROR_THROW_ON(
            BINEIntentFFNLayer::validate(input->info(),
                fc_weights->info(),
                fc_bias->info(),
                proj_weights->info(),
                proj_bias->info(),
                gamma->info(),
                output->info()));

        BI_COMPUTE_LOG_PARAMS(input, fc_weights, fc_bias, proj_weights, proj_bias, gamma, act_info, output); // 获取log的参数

        _max_batch = max_batch_size;
        _max_seq = max_seq_len;
        _seq_len = cur_seq_size;
        _batch_size = cur_batch_size;

        // 中间变量输出的形状
        BITensorShape norm_output_shape = BITensorShape(768, _max_seq, _max_batch); // 归一化输出
        BITensorShape fc_fuse_output_shape = BITensorShape(3072, _max_seq, _max_batch); // Gemm + GeLU 融合操作
        BITensorShape proj_output_shape = BITensorShape(768, _max_seq, _max_batch); // 最后降解的操作

        // 初始化中间变量
        _norm_output.allocator()->init(BITensorInfo(norm_output_shape, 1, BIDataType::F16));
        _fuse_output.allocator()->init(BITensorInfo(fc_fuse_output_shape, 1, BIDataType::F16));
        _proj_output.allocator()->init(BITensorInfo(proj_output_shape, 1, BIDataType::F16));

        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_fuse_output);
        _memory_group.manage(&_proj_output);

        _norm_output.allocator()->allocate();
        _fuse_output.allocator()->allocate();
        _proj_output.allocator()->allocate();

        const auto sub_norm_output_shape = BITensorShape(768, _seq_len, _batch_size);
        _sub_norm_output_info = BITensorInfo(sub_norm_output_shape, 1, BIDataType::F16);
        _sub_norm_output_info.set_format(Format::F16);
        _sub_norm_output.allocator()->init(_sub_norm_output_info);

        const auto sub_fc_fuse_output_shape = BITensorShape(3072, _seq_len, _batch_size);
        _sub_fuse_output_info = BITensorInfo(sub_fc_fuse_output_shape, 1, BIDataType::F16);
        _sub_fuse_output_info.set_format(Format::F16);
        _sub_fuse_output.allocator()->init(_sub_fuse_output_info);

        _sub_proj_output_info = BITensorInfo(sub_norm_output_shape, 1, BIDataType::F16);
        _sub_proj_output_info.set_format(Format::F16);
        _sub_proj_output.allocator()->init(_sub_proj_output_info);

        // 配置GemmInfo
        GEMMInfo fc_gemm_info, proj_gemm_info;
        fc_gemm_info.set_activation_info(act_info);
        fc_gemm_info.set_fast_math(true);
        proj_gemm_info.set_fast_math(true);

        // 算子进行配置
        _layer_norm_layer.configure(input, gamma, ln_2_bias, &_sub_norm_output);
        _c_fc_fuse_act.configure(&_sub_norm_output, fc_weights, fc_bias, &_sub_fuse_output, 1.f, 1.f, fc_gemm_info);
        _c_proj.configure(&_sub_fuse_output, proj_weights, proj_bias, &_sub_proj_output, 1.0f, 1.0f, proj_gemm_info);
        _copy_f.configure(&_sub_proj_output, output);
    }

    void BINEIntentFFNLayer::run() {
        prepare();
        _layer_norm_layer.run();
        _c_fc_fuse_act.run();
        _c_proj.run();
        _copy_f.run();
    }

    void BINEIntentFFNLayer::prepare() {
        if (!_is_prepared) {
            _scope_mg = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);
            _sub_norm_output.allocator()->init(*_norm_output.allocator(), _sub_norm_output_info);
            _sub_fuse_output.allocator()->init(*_fuse_output.allocator(), _sub_fuse_output_info);
            _sub_proj_output.allocator()->init(*_proj_output.allocator(), _sub_proj_output_info);
            _is_prepared = true;
        }
    }
}