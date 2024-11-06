//
// Created by Mason on 2024/10/16.
//
#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace BatmanInfer {
    RuntimeGraph::RuntimeGraph(std::string model_path) : model_path_(model_path) {

    }


    const std::string &RuntimeGraph::model_path() {
        return this->model_path_;
    }

    void RuntimeGraph::set_model_path(const std::string& model_path) {
        this->model_path_ = model_path;
    }

    bool RuntimeGraph::Init() {
        if (this->model_path_.empty()) {
            LOG(ERROR) << "The model path is empty";
            return false;
        }

        this->graph_ = std::make_unique<ONNXGraph>();
        int load_result = this->graph_->load(model_path_);
        if (load_result != 0) {
            LOG(ERROR) << "Can not find the model path: " << model_path_;
            return false;
        }

        std::vector<ONNXOperator *> operators = this->graph_->operators;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        this->operators_.clear();
        this->operators_maps_.clear();
        for (const ONNXOperator *op: operators) {
            if (!op) {
                LOG(ERROR) << "Meet the empty node";
                continue;
            } else {
                std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
                // 初始化算子的名称
                runtime_operator->name = op->name;
                runtime_operator->type = op->type;

                // 初始化算子中的input
                const std::vector<ONNXOperand *> &inputs = op->inputs;
                if (!inputs.empty()) {
                    InitGraphOperatorsInput(inputs, runtime_operator);
                }

                // 记录输出operand中的名称
                const std::vector<ONNXOperand *> &outputs = op->outputs;
                if (!outputs.empty()) {
                    InitGraphOperatorsOutput(outputs, runtime_operator);
                }

                // 初始化算子中的attribute(权重)
                const std::map<std::string, ONNXAttribute> &attrs = op->attrs;
                if (!attrs.empty())
                    InitGraphAttrs(attrs, runtime_operator);

                // 初始化算子中的Parameter
                const std::map<std::string, ONNXParameter> &params = op->params;
                if (!params.empty())
                    InitGraphParams(params, runtime_operator);

                this->operators_.push_back(runtime_operator);
                this->operators_maps_.insert({runtime_operator->name,
                                              runtime_operator});

            }
        }

        graph_state_ = GraphState::NeedBuild;
        return true;
    }

    void RuntimeGraph::InitGraphOperatorsInput(const std::vector<ONNXOperand *> &inputs,
                                               const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const ONNXOperand *input: inputs) {
            if (!input) {
                continue;
            }
            const ONNXOperator *producer = input->producer;
            std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
            runtime_operand->name = producer->name;
            runtime_operand->shapes = input->shape;

            switch (input->type) {
                case 1:
                    runtime_operand->type = RuntimeDataType::kTypeFloat32;
                    break;
                case 0:
                    runtime_operand->type = RuntimeDataType::kTypeUnknown;
                    break;
                default:
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
            }
            runtime_operator->input_operands.insert({producer->name, runtime_operand});
            runtime_operator->input_operands_seq.push_back(runtime_operand);
        }
    }

    void RuntimeGraph::InitGraphOperatorsOutput(const std::vector<ONNXOperand *> &outputs,
                                                const std::shared_ptr<RuntimeOperator> &runtime_operator) {
         for (const ONNXOperand *output: outputs) {
             if (!output)
                 continue;
             const auto& consumers = output->consumers;
             for (const auto& c : consumers)
                 runtime_operator->output_names.push_back(c->name);
         }
    }

    void RuntimeGraph::InitGraphAttrs(const std::map<std::string, ONNXAttribute> &attrs,
                                      const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &[name, attr]: attrs) {
            switch (attr.type) {
                case 1: {
                    std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
                    runtime_attribute->type = RuntimeDataType::kTypeFloat32;
                    runtime_attribute->weight_data = attr.data;
                    runtime_attribute->shape = attr.shape;
                    runtime_operator->attribute.insert({name, runtime_attribute});
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown attribute type: " << attr.type;
                }
            }
        }
    }

    void RuntimeGraph::InitGraphParams(const std::map<std::string, ONNXParameter> &params,
                                       const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &[name, parameter]: params) {
            const int type = parameter.type;
            switch (type) {
                case int(RuntimeParameterType::bParameterUnknown): {
                    std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                case int(RuntimeParameterType::bParameterBool): {
                    std::shared_ptr<RuntimeParameterBool> runtime_parameter = std::make_shared<RuntimeParameterBool>();
                    runtime_parameter->value = parameter.b;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::bParameterInt): {
                    std::shared_ptr<RuntimeParameterInt> runtime_parameter = std::make_shared<RuntimeParameterInt>();
                    runtime_parameter->value = parameter.i;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::bParameterFloat): {
                    std::shared_ptr<RuntimeParameterFloat> runtime_parameter = std::make_shared<RuntimeParameterFloat>();
                    runtime_parameter->value = parameter.f;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::bParameterString): {
                    std::shared_ptr<RuntimeParameterString> runtime_parameter = std::make_shared<RuntimeParameterString>();
                    runtime_parameter->value = parameter.s;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::bParameterIntArray): {
                    std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
                            std::make_shared<RuntimeParameterIntArray>();
                    runtime_parameter->value = parameter.ai;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::bParameterFloatArray): {
                    std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
                            std::make_shared<RuntimeParameterFloatArray>();
                    runtime_parameter->value = parameter.af;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                case int(RuntimeParameterType::bParameterStringArray): {
                    std::shared_ptr<RuntimeParameterStringArray> runtime_parameter = std::make_shared<RuntimeParameterStringArray>();
                    runtime_parameter->value = parameter.as;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown parameter type: " << type;
                }
            }
        }
    }

    void RuntimeGraph::ReverseToPo(const std::shared_ptr<RuntimeOperator> &root_op) {
        CHECK(root_op != nullptr) << "Current operator is nullptr";
        root_op->has_forward = true;
        // 获取后面的算子
        const auto &next_ops = root_op->output_operators;
        for (const auto &[_, op]: next_ops) {
            if (op != nullptr) {
                // 如果没有前面的算子
                if (!op->has_forward)
                    this->ReverseToPo(op);
            }
        }
        for (const auto &[_, op]: next_ops)
            CHECK_EQ(op->has_forward, true);
        this->to_po_operators_.push_back(root_op);
    }

    void RuntimeGraph::Build(const std::string &input_name, const std::string &output_name) {
        if (graph_state_ == GraphState::Complete) {
            LOG(INFO) << "Model has been built already!";
            return;
        }

        if (graph_state_ == GraphState::NeedInit) {
            bool init_graph = Init();
            LOG_IF(FATAL, !init_graph) << "Init graph failed!";
        }

        CHECK(graph_state_ >= GraphState::NeedBuild)
             << "Graph status error, current state is " << int(graph_state_);
        LOG_IF(FATAL, this->operators_.empty())
             << "Graph operators is empty, may can not be init";

        // 构建图关系
        for (const auto &current_op : this->operators_) {
            // 获取当前节点的所有后继节点的names, 遍历根据next_op_name从operators_maps_中插入所需要的结点
            const std::vector<std::string> &output_names = current_op->output_names;
            for (const auto &b_output_name: output_names) {
                if (const auto &output_op = this->operators_maps_.find(b_output_name);
                    output_op != this->operators_maps_.end())
                    current_op->output_operators.insert({b_output_name, output_op->second});
            }
        }

        // 初始化结点的输入和输出空间
        RuntimeOperatorUtils::InitOperatorInput(operators_);
        RuntimeOperatorUtils::InitOperatorOutput(graph_->operators, operators_);

        // 构建拓扑顺序
        to_po_operators_.clear();
        for (const auto &[_, op] : operators_maps_) {
            // 根据输入节点构建拓扑排序
            if (op->type == "Input" && !op->has_forward) {
                this->ReverseToPo(op);
            }
        }

        CHECK(to_po_operators_.size() == operators_.size()) << "Build wrong to_po queue";
        std::reverse(to_po_operators_.begin(), to_po_operators_.end());

        graph_state_ = GraphState::Complete;
        input_name_ = input_name;
        output_name_ = output_name;
        if (graph_ != nullptr) {
            graph_.reset();
            graph_ = nullptr;
        }
    }

    RuntimeGraph::GraphState RuntimeGraph::graph_state() const {
        return this->graph_state_;
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> &
    RuntimeGraph::get_to_po_queues() const {
        return this->to_po_operators_;
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::operators() const {
        return this->operators_;
    }
}