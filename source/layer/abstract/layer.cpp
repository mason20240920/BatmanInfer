//
// Created by Mason on 2024/10/29.
//

#include <layer/abstract/layer.hpp>
#include <glog/logging.h>

namespace BatmanInfer {
    const std::shared_ptr<Tensor<float>>& Layer::weights() const {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    const std::shared_ptr<Tensor<float>>& Layer::bias() const {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_bias(const std::vector<float> &bias) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_weights(const std::vector<float> &weights) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_bias(const std::shared_ptr<Tensor<float>> &bias) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_weights(const std::shared_ptr<Tensor<float>> &weights) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    InferStatus Layer::Forward(const std::map<std::string , std::shared_ptr<Tensor<float>>>& inputs,
                               std::map<std::string, std::shared_ptr<Tensor<float>>>& outputs){
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    InferStatus Layer::Forward() {
        LOG_IF(FATAL, this->runtime_operator_.expired())
              << "Runtime operator is expired or nullptr";
        const auto& runtime_operator = this->runtime_operator_.lock();
        // 准备节点Layer计算所需的输入
        const std::vector<std::shared_ptr<RuntimeOperand>>& input_operand_lst =
                runtime_operator->input_operands_seq;
        // layer的输入, 每层都是一个map文件，对应的是
        std::map<std::string, std::shared_ptr<Tensor<float>>> layer_input_map;
        for (const auto& input_operand_data : input_operand_lst) {
            for (const auto& input_data : input_operand_lst)
                layer_input_map[runtime_operator->name] = input_data->data;
        }

        // layer的输出
        const auto& output_operand_lst = runtime_operator->output_operands;
        // 确保 output_operands 不为空
        CHECK(!output_operand_lst.empty()) << "runtime_operator->output_operands is empty";
        // 创建 layer_output_data，并确保与 output_operands 的 data 共享内存
        std::map<std::string , std::shared_ptr<Tensor<float>>> layer_output_map;

        // 遍历 map，将每个 output_operand 的 data 添加到 layer_output_data 中
        for (auto& [name, operand] : output_operand_lst) {
            // 直接插入 data 的元素到 layer_output_data 中
            layer_input_map[name] = operand->data;
        }


        if (runtime_operator->type != "Constant")
            CHECK(!layer_input_map.empty())
                            << runtime_operator->name << "Layer input data is empty";
        CHECK(!layer_output_map.empty())
             << "Layer output data is empty";

        // 执行operator 当中的Layer计算过程
        InferStatus status = runtime_operator->layer->Forward(
                layer_input_map, layer_output_map);

        // 根据map进行重新赋值
        for ( auto& [key, value]: output_operand_lst) {
            value->data = layer_output_map[key];
        }
        return status;
    }

    void Layer::set_runtime_operator(const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        CHECK(runtime_operator != nullptr);
        this->runtime_operator_ = runtime_operator;
    }
}