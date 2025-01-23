//
// Created by Mason on 2025/1/23.
//

#include <runtime/neon/functions/bi_ne_rnn_layer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>

namespace BatmanInfer {
    BINERNNLayer::~BINERNNLayer() = default;

    BINERNNLayer::BINERNNLayer(std::shared_ptr<BIIMemoryManager> memory_manager) :
            _memory_group(std::move(memory_manager)),
            _gemm_state_f(),
            _add_f(),
            _activation(),
            _fully_connected(memory_manager),
            _copy_f(),
            _fully_connected_out(),
            _gemm_output(),
            _add_output(),
            _is_prepared(false) {

    }

    BIStatus BINERNNLayer::validate(const BatmanInfer::BIITensorInfo *input,
                                    const BatmanInfer::BIITensorInfo *weights,
                                    const BatmanInfer::BIITensorInfo *recurrent_weights,
                                    const BatmanInfer::BIITensorInfo *bias,
                                    const BatmanInfer::BIITensorInfo *hidden_state,
                                    const BatmanInfer::BIITensorInfo *output,
                                    const BatmanInfer::BIActivationLayerInfo &info) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, recurrent_weights, bias, hidden_state, output);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, BIDataType::F16, BIDataType::F32);

        const int idx_width = get_data_layout_dimension_index(input->data_layout(), BIDataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(input->data_layout(), BIDataLayoutDimension::HEIGHT);
        BI_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_width) != weights->dimension(idx_width));
        BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 2);
        BI_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_height) != recurrent_weights->dimension(idx_width));
        BI_COMPUTE_RETURN_ERROR_ON(
                recurrent_weights->dimension(idx_width) != recurrent_weights->dimension(idx_height));
        BI_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() != 1);
        BI_COMPUTE_RETURN_ERROR_ON(bias->dimension(idx_width) != weights->dimension(idx_height));
        BI_COMPUTE_RETURN_ERROR_ON(hidden_state->dimension(idx_width) != weights->dimension(idx_height));
        BI_COMPUTE_RETURN_ERROR_ON(hidden_state->dimension(idx_height) != input->dimension(idx_height));
        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), hidden_state->tensor_shape());

        auto shape_info = BITensorInfo(misc::shape_calculator::compute_rnn_shape(recurrent_weights,
                                                                                 hidden_state->dimension(idx_height)),
                                       1,
                                       input->data_type());

        BI_COMPUTE_RETURN_ON_ERROR(BINEFullyConnectedLayer::validate(input, weights, bias, &shape_info));
        BI_COMPUTE_RETURN_ON_ERROR(
                BINEArithmeticAddition::validate(&shape_info, &shape_info, &shape_info, BIConvertPolicy::SATURATE));
        BI_COMPUTE_RETURN_ON_ERROR(BINEActivationLayer::validate(&shape_info, &shape_info, info));

        return BIStatus{};
    }

    void BINERNNLayer::configure(const BatmanInfer::BIITensor *input,
                                 const BatmanInfer::BIITensor *weights,
                                 const BatmanInfer::BIITensor *recurrent_weights,
                                 const BatmanInfer::BIITensor *bias,
                                 BatmanInfer::BIITensor *hidden_state,
                                 BatmanInfer::BIITensor *output,
                                 BatmanInfer::BIActivationLayerInfo &info) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, weights, recurrent_weights, bias, hidden_state, output);
        BI_COMPUTE_ERROR_THROW_ON(BINERNNLayer::validate(input->info(), weights->info(), recurrent_weights->info(),
                                                         bias->info(), hidden_state->info(), output->info(), info));
        BI_COMPUTE_LOG_PARAMS(input, weights, recurrent_weights, bias, hidden_state, output, info);

        // 获取输入张量的高度索引，根据数据布局（NHWC或NCHW）动态计算
        const int idx_height = get_data_layout_dimension_index(input->info()->data_layout(),
                                                               BIDataLayoutDimension::HEIGHT);
        // 计算RNN的中间张量形状，基于循环权重和隐藏状态高度
        BITensorShape shape = misc::shape_calculator::compute_rnn_shape(recurrent_weights->info(),
                                                                        hidden_state->info()->dimension(idx_height));

        // 初始化标志, 标识尚未准备好
        _is_prepared = false;

        // 初始化中间张量 _fully_connected_out, 用于存储全连接层输出
        _fully_connected_out.allocator()->init(BITensorInfo(shape, 1, input->info()->data_type()));

        // 初始化中间张量 _gemm_output, 用于存储GEMM输出
        _gemm_output.allocator()->init(BITensorInfo(shape, 1, input->info()->data_type()));

        // 将_fully_connected_out张量教由内存管理器管理
        _memory_group.manage(&_fully_connected_out);

        // 配置全连接层，计算输入张量、权重和偏置的线性变换，输出到 _fully_connected_out
        _fully_connected.configure(input, weights, bias, &_fully_connected_out);

        // 将 _gemm_output 张量交由内存管理器管理
        _memory_group.manage(&_gemm_output);

        // 配置 GEMM 操作，用于计算隐藏状态和循环权重的线性变换
        _gemm_state_f.configure(hidden_state, recurrent_weights, nullptr, &_gemm_output, 1.f, 0.f);

        // 初始化中间张量 _add_output，用于存储全连接输出和 GEMM 输出相加后的结果
        _add_output.allocator()->init(BITensorInfo(shape, 1, input->info()->data_type()));

        // 将 _add_output 张量交由内存管理器管理
        _memory_group.manage(&_add_output);

        // 配置加法操作，将 _fully_connected_out 和 _gemm_output 相加，结果存储在 _add_output 中
        _add_f.configure(&_fully_connected_out, &_gemm_output, &_add_output, BIConvertPolicy::SATURATE);

        // 为 _fully_connected_out 和 _gemm_output 分配内存
        _fully_connected_out.allocator()->allocate();
        _gemm_output.allocator()->allocate();

        // 配置激活函数，将 _add_output 作为输入，结果存储在 hidden_state 中
        _activation.configure(&_add_output, hidden_state, info);

        // 为 _add_output 分配内存
        _add_output.allocator()->allocate();

        // 配置复制操作，将 hidden_state 的内容复制到 output 中
        _copy_f.configure(hidden_state, output);
    }

    void BINERNNLayer::run() {
        prepare();

        BIMemoryGroupResourceScope scope_mg(_memory_group);

        _fully_connected.run();

        _gemm_state_f.run();

        _add_f.run();
        _activation.run();

        // copy hidden out to output
        _copy_f.run();
    }

    void BINERNNLayer::prepare() {
        if (!_is_prepared) {
            _fully_connected.prepare();
            _gemm_state_f.prepare();

            _is_prepared = true;
        }
    }
}