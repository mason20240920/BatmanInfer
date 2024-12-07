//
// Created by Mason on 2024/10/29.
//

#include <layer/abstract/layer.hpp>
#include <glog/logging.h>

namespace BatmanInfer {
    const std::vector<std::shared_ptr<Tensor<float>>>& Layer::weights() const {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    const std::vector<std::shared_ptr<Tensor<float>>>& Layer::bias() const {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_bias(const std::vector<float> &bias) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_weights(const std::vector<float> &weights) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    InferStatus Layer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    InferStatus Layer::Forward() {
        LOG_IF(FATAL, this->runtime_operator_.expired())
              << "Runtime operator is expired or nullptr";
        const auto& runtime_operator = this->runtime_operator_.lock();
        // 准备节点Layer计算所需的输入
        const std::vector<std::shared_ptr<RuntimeOperand>>& input_operand_datas =
                runtime_operator->input_operands_seq;
        // layer的输入
        std::vector<std::shared_ptr<Tensor<float>>> layer_input_data;
        for (const auto& input_operand_data : input_operand_datas) {
            for (const auto& input_data : input_operand_data->datas)
                layer_input_data.push_back(input_data);
        }

        // layer的输出
        const auto& output_operand_lst = runtime_operator->output_operands;
        // 确保 output_operands 不为空
        CHECK(!output_operand_lst.empty()) << "runtime_operator->output_operands is empty";
        // 创建 layer_output_data，并确保与 output_operands 的 datas 共享内存
        std::vector<std::shared_ptr<Tensor<float>>> layer_output_data;

        // 遍历 map，将每个 output_operand 的 datas 添加到 layer_output_data 中
        for (auto& [name, operand] : output_operand_lst) {
            // 直接插入 datas 的元素到 layer_output_data 中
            layer_output_data.insert(
                    layer_output_data.end(),
                    operand->datas.begin(),
                    operand->datas.end()
            );
        }


        if (runtime_operator->type != "Constant")
            CHECK(!layer_input_data.empty())
                            << runtime_operator->name << "Layer input data is empty";
        CHECK(!layer_output_data.empty())
             << "Layer output data is empty";

        // 执行operator 当中的Layer计算过程
        InferStatus status = runtime_operator->layer->Forward(
                layer_input_data, layer_output_data);

        // 根据map进行重新赋值
        int output_index = 0;
        for ( auto& [key, value]: output_operand_lst) {
            value->datas = std::vector<sftensor>{layer_output_data[output_index]};
            output_index += 1;
        }
        return status;
    }

    void Layer::set_runtime_operator(const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        CHECK(runtime_operator != nullptr);
        this->runtime_operator_ = runtime_operator;
    }
}