//
// Created by Mason on 2025/1/22.
//

#include <runtime/neon/functions/bi_NEFullyConnectedLayer.h>

#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/neon/functions/bi_ne_convert_fully_connected_weights.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/operators/bi_cpu_fully_connected.hpp>

namespace BatmanInfer {
    using namespace BatmanInfer::experimental;

    struct BINEFullyConnectedLayer::Impl {
        // 用于管理内存分组的对象
        BIMemoryGroup memory_group{};
        // 权重管理器指针
        BIIWeightsManager *weights_manager{nullptr};

        // BICpuFullyConnected 的独占指针
        std::unique_ptr<cpu::BICpuFullyConnected> op{nullptr};

        // 原始权重张量指针
        const BIITensor *original_weights{nullptr};

        BIITensorPack run_pack{};  // 用于运行时的张量包
        WorkspaceData<BITensor> workspace{}; // 工作空间数据
        experimental::BIMemoryRequirements aux_mem_req{};  // 辅助内存需求

        bool is_prepared{false};           // 标记是否已准备好
        bool dynamic_weights{false};       // 标记权重是否为动态权重
    };

    BINEFullyConnectedLayer::~BINEFullyConnectedLayer() = default;

    BINEFullyConnectedLayer::BINEFullyConnectedLayer(std::shared_ptr<BIIMemoryManager> memory_manager,
                                                     BatmanInfer::BIIWeightsManager *weights_manager) :
            _impl(std::make_unique<Impl>()) {
        // 初始化内存组
        _impl->memory_group = BIMemoryGroup(std::move(memory_manager));
        // 初始化权重管理器
        _impl->weights_manager = weights_manager;
    }

    void BINEFullyConnectedLayer::configure(const BatmanInfer::BIITensor *input, const BatmanInfer::BIITensor *weights,
                                            const BatmanInfer::BIITensor *biases, BatmanInfer::BIITensor *output,
                                            BatmanInfer::BIFullyConnectedLayerInfo fc_info,
                                            const BatmanInfer::BIWeightsInfo &weights_info) {
        // 验证输入参数是否为非空
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        // 执行验证步骤，检查输入参数是否合法
        BI_COMPUTE_ERROR_THROW_ON(BINEFullyConnectedLayer::validate(input->info(), weights->info(),
                                                                    biases != nullptr ? biases->info() : nullptr,
                                                                    output->info(), fc_info, weights_info));

        // 记录参数信息（用于调试或日志记录）
        BI_COMPUTE_LOG_PARAMS(input, weights, biases, output, fc_info);

        // 创建 CpuFullyConnected 实例，并保存到 op 成员变量
        _impl->op = std::make_unique<cpu::BICpuFullyConnected>();
        // 保存原始权重张量的指针
        _impl->original_weights = weights;
        // 标记为未准备状态
        _impl->is_prepared = false;

        // 配置 CpuFullyConnected 实例，传递输入、权重、偏置、输出张量的信息
        _impl->op->configure(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr,
                             output->info(),
                             fc_info, weights_info);

        // 如果权重管理器存在，则管理原始权重张量
        if (_impl->weights_manager != nullptr)
            _impl->weights_manager->manage(_impl->original_weights);

        // 获取操作的辅助内存需求
        _impl->aux_mem_req = _impl->op->workspace();
        // 初始化运行时张量包，包含输入、权重、偏置和输出张量
        _impl->run_pack = {{ACL_SRC_0, input},
                           {ACL_SRC_1, weights},
                           {ACL_SRC_2, biases},
                           {ACL_DST,   output}};
        // 管理辅助内存工作空间
        _impl->workspace = manage_workspace<BITensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack,
                                                      _impl->run_pack, /* allocate_now */ false);

        // 检查权重是否为动态权重，根据权重特性和配置信息设置 dynamic_weights 标志
        _impl->dynamic_weights = !weights->info()->are_values_constant() && fc_info.transpose_weights &&
                                 !fc_info.are_weights_reshaped && !fc_info.retain_internal_weights;
    }

    // 检查是否存在优化实现
    BIStatus BINEFullyConnectedLayer::has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                                                   const BIITensorInfo *input,
                                                   const BIITensorInfo *weights,
                                                   const BIITensorInfo *biases,
                                                   const BIITensorInfo *output,
                                                   const BIFullyConnectedLayerInfo &fc_info,
                                                   const BIWeightsInfo &weights_info) {
        return cpu::BICpuFullyConnected::has_opt_impl(expected_weight_format, input, weights, biases, output, fc_info,
                                                      weights_info);
    }

    // 验证函数，检查输入参数是否符合要求
    BIStatus BINEFullyConnectedLayer::validate(const BIITensorInfo *input,
                                               const BIITensorInfo *weights,
                                               const BIITensorInfo *biases,
                                               const BIITensorInfo *output,
                                               BIFullyConnectedLayerInfo fc_info,
                                               const BIWeightsInfo &weights_info) {
        return cpu::BICpuFullyConnected::validate(input, weights, biases, output, fc_info, weights_info);
    }

    // 运行函数，执行全连接层的计算
    void BINEFullyConnectedLayer::run() {
        // 如果权重不是动态的，则提前调用 prepare 函数
        if (!_impl->dynamic_weights) {
            prepare();
        }

        // 创建内存组资源作用域，确保内存资源在作用域内被管理
        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        // 调用 CpuFullyConnected 实例的运行函数，执行实际计算
        _impl->op->run(_impl->run_pack);
    }

    // 准备函数，执行必要的准备工作，例如分配临时内存、预处理权重等
    void BINEFullyConnectedLayer::prepare() {
        // 如果尚未准备好，则执行准备工作
        if (!_impl->is_prepared) {
            // 分配辅助内存
            allocate_tensors(_impl->aux_mem_req, _impl->workspace);
            // 创建内存组资源作用域
            BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
            // 调用 CpuFullyConnected 实例的 prepare 函数
            _impl->op->prepare(_impl->run_pack);

            // 释放仅在 prepare 阶段使用的临时张量
            release_temporaries<BITensor>(_impl->aux_mem_req, _impl->workspace);
            // 标记为已准备状态
            _impl->is_prepared = true;

            // 如果权重管理器存在且权重已被管理，则处理权重释放逻辑
            if (_impl->weights_manager != nullptr &&
                _impl->weights_manager->are_weights_managed(_impl->original_weights)) {
                // 确保权重在最后一次引用完成后才被释放
                const BIITensor *original_b = _impl->original_weights;
                if (!original_b->is_used()) {
                    _impl->weights_manager->pre_mark_as_unused(original_b);
                }
                _impl->original_weights->mark_as_used();
                _impl->weights_manager->release(_impl->original_weights);
            }
        }
    }
}