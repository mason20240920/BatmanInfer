//
// Created by Mason on 2025/2/10.
//

#include <runtime/neon/functions/BINEFeedForwardLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

namespace BatmanInfer {
    BINEFeedForwardLayer::~BINEFeedForwardLayer() = default;

    BINEFeedForwardLayer::BINEFeedForwardLayer(std::shared_ptr<BIIMemoryManager> memory_manager) :
            _memory_group(std::move(memory_manager)),
            _normalization_layer(),
            _c_fc_fuse_act(),
            _c_proj(),
            _copy_f(),
            _norm_output(),
            _fuse_output(),
            _proj_output(),
            _is_prepared(false) {

    }

    BIStatus
    BINEFeedForwardLayer::validate(const BatmanInfer::BIITensorInfo *input,
                                   const BatmanInfer::BIITensorInfo *fc_weights,
                                   const BatmanInfer::BIITensorInfo *fc_bias,
                                   const BatmanInfer::BIITensorInfo *proj_weights,
                                   const BatmanInfer::BIITensorInfo *proj_bias,
                                   const BatmanInfer::BIITensorInfo *output) {

        BI_COMPUTE_ERROR_ON_NULLPTR(input, fc_weights, fc_bias, proj_weights, proj_bias, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F16, BIDataType::F32);

        BI_COMPUTE_RETURN_ERROR_ON(input->tensor_shape() != output->tensor_shape());

        return BIStatus{};

    }

    void BINEFeedForwardLayer::configure(const BatmanInfer::BIITensor *input, const BatmanInfer::BIITensor *fc_weights,
                                         const BatmanInfer::BIITensor *fc_bias,
                                         const BatmanInfer::BIITensor *proj_weights,
                                         const BatmanInfer::BIITensor *proj_bias,
                                         const BatmanInfer::BIActivationLayerInfo &act_info,
                                         const BINormalizationLayerInfo &norm_info,
                                         BatmanInfer::BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, fc_weights, fc_bias, proj_weights, proj_bias, output); // 输入的参数是否为空

        BI_COMPUTE_ERROR_THROW_ON(
                BINEFeedForwardLayer::validate(input->info(),
                                               fc_weights->info(),
                                               fc_bias->info(),
                                               proj_weights->info(),
                                               proj_bias->info(),
                                               output->info()));

        BI_COMPUTE_LOG_PARAMS(input, fc_weights, fc_bias, proj_weights, proj_bias, act_info, norm_info,
                              output); // 获取log的参数

        // 中间变量输出的形状
        BITensorShape norm_output_shape = BITensorShape(input->info()->tensor_shape()); // 归一化输出
        BITensorShape fc_fuse_output_shape = BITensorShape(3072, 16); // Gemm + GeLU 融合操作
        BITensorShape proj_output_shape = BITensorShape(output->info()->tensor_shape()); // 最后降解的操作

        // 初始化中间变量
        _norm_output.allocator()->init(BITensorInfo(norm_output_shape, 1, input->info()->data_type()));
        _fuse_output.allocator()->init(BITensorInfo(fc_fuse_output_shape, 1, input->info()->data_type()));
        _proj_output.allocator()->init(BITensorInfo(proj_output_shape, 1, input->info()->data_type()));

        // 内存管理
        _memory_group.manage(&_norm_output);
        _memory_group.manage(&_fuse_output);
        _memory_group.manage(&_proj_output);

        _norm_output.allocator()->allocate();
        _fuse_output.allocator()->allocate();
        _proj_output.allocator()->allocate();

        // 配置GemmInfo
        GEMMInfo fc_gemm_info, proj_gemm_info;
        fc_gemm_info.set_activation_info(act_info);
        fc_gemm_info.set_fast_math(true);
        proj_gemm_info.set_fast_math(true);

        // 算子进行配置
        _normalization_layer.configure(input, &_norm_output, norm_info);
        _c_fc_fuse_act.configure(&_norm_output, fc_weights, fc_bias, &_fuse_output, 1.f, 1.f, fc_gemm_info);
        _c_proj.configure(&_fuse_output, proj_weights, proj_bias, &_proj_output, 1.0f, 1.0f, proj_gemm_info);
        _copy_f.configure(&_proj_output, output);
    }

    void BINEFeedForwardLayer::run() {
        prepare();
        BIMemoryGroupResourceScope scope_mg(_memory_group);
        _normalization_layer.run();
        _c_fc_fuse_act.run();
        _c_proj.run();
        _copy_f.run();
    }

    void BINEFeedForwardLayer::prepare() {
        if (!_is_prepared) {
//            _reshape.prepare();
//            _gemm_state_f.prepare();

            _is_prepared = true;
        }
    }
}