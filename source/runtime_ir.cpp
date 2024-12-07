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
#include <layer/abstract/layer.hpp>
#include <layer/abstract/layer_factory.hpp>

namespace BatmanInfer {
    RuntimeGraph::RuntimeGraph(std::string model_path) : model_path_(std::move(model_path)) {
    }


    const std::string &RuntimeGraph::model_path() {
        return this->model_path_;
    }

    void RuntimeGraph::set_model_path(const std::string &model_path) {
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
                this->operators_maps_.insert({
                    runtime_operator->name,
                    runtime_operator
                });
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
                case 2: {
                    runtime_operand->type = RuntimeDataType::kTypeFloat64;
                    break;
                }
                case 5:
                    runtime_operand->type = RuntimeDataType::kTypeInt64;
                    break;
                case 9:
                    runtime_operand->type = RuntimeDataType::kTypeBoolean;
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
            const auto &consumers = output->consumers;
            for (const auto &c: consumers) {
                runtime_operator->output_names.push_back(c->name);
                std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
                runtime_operand->name = c->name;
                runtime_operator->output_operands.insert({c->name, runtime_operand});
            }
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
                case 5: {
                    std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
                    runtime_attribute->type = RuntimeDataType::kTypeInt64;
                    runtime_attribute->weight_data = attr.data;
                    runtime_attribute->shape = attr.shape;
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
                    std::shared_ptr<RuntimeParameterFloat> runtime_parameter = std::make_shared<
                        RuntimeParameterFloat>();
                    runtime_parameter->value = parameter.f;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::bParameterString): {
                    std::shared_ptr<RuntimeParameterString> runtime_parameter = std::make_shared<
                        RuntimeParameterString>();
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
                    std::shared_ptr<RuntimeParameterStringArray> runtime_parameter = std::make_shared<
                        RuntimeParameterStringArray>();
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

    std::shared_ptr<Layer> RuntimeGraph::CreateLayer(const std::shared_ptr<RuntimeOperator> &op) {
        LOG_IF(FATAL, !op) << "Operator is empty!";
        auto layer = LayerRegister::CreateLayer(op);
        LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
        return layer;
    }

    // 此函数不关心用户使用 Build 函数时提供的 input 列表。它将所有的 input 与 output 都构建进图中
    void RuntimeGraph::TopoSortOperators() {
        to_po_operators_.clear();
        to_po_operators_.reserve(this->operators_.size());
        std::map<std::shared_ptr<RuntimeOperator>, int> op_in_degrees;
        std::queue<std::shared_ptr<RuntimeOperator> > zero_degree_queue;

        for (auto &op: this->operators_) {
            if ("Input" == op->type || "Constant" == op->type) {
                zero_degree_queue.push(op);
            } else {
                op_in_degrees.insert(std::make_pair(op, op->input_operands_seq.size()));
            }
        }

        while (!zero_degree_queue.empty()) {
            auto op = zero_degree_queue.front();
            zero_degree_queue.pop();
            this->to_po_operators_.push_back(op);

            const auto &next_ops = op->output_operators;
            for (const auto &[_, sub_op]: next_ops) {
                auto iter = op_in_degrees.find(sub_op);
                CHECK(iter != op_in_degrees.end());
                CHECK(iter->second > 0);

                --iter->second;
                if (0 == iter->second) {
                    zero_degree_queue.push(sub_op);
                }
            }
        }
    }


    void RuntimeGraph::Build(const std::vector<std::string> &input_names_strings,
                             const std::vector<std::string> &output_names_strings) {
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
        for (const auto &current_op: this->operators_) {
            // 获取当前节点的所有后继节点的names, 遍历根据next_op_name从operators_maps_中插入所需要的结点
            const std::vector<std::string> &output_names = current_op->output_names;
            for (const auto &b_output_name: output_names) {
                if (const auto &output_op = this->operators_maps_.find(b_output_name);
                    output_op != this->operators_maps_.end())
                    current_op->output_operators.insert({b_output_name, output_op->second});
            }
        }

        for (const auto &kOperator: this->operators_) {
            // 除了输入和输出节点，都创建Layer
            if (kOperator->type != "Input" && kOperator->type != "Output") {
                std::shared_ptr<Layer> layer = RuntimeGraph::CreateLayer(kOperator);
                CHECK(layer != nullptr) << "Layer " << kOperator->name << " create failed!";
                if (layer) {
                    kOperator->layer = layer;
                    layer->set_runtime_operator(kOperator);
                }
            }
        }

        // 初始化结点的输入和输出空间
        RuntimeOperatorUtils::InitOperatorInput(operators_);
        RuntimeOperatorUtils::InitOperatorOutput(graph_->operators, operators_);

        // 检查输入输出是否合法，只会检查提供的输入输出中是否有模型不能识别的
        input_names_ = input_names_strings;
        output_names_ = output_names_strings;
        std::set<std::string> check_input(input_names_.begin(), input_names_.end());
        std::set<std::string> check_output(output_names_.begin(), output_names_.end());
        for (const auto &op: this->operators_) {
            if ("Input" == op->type) {
                if (auto iter = check_input.find(op->name); iter != check_input.end()) {
                    check_input.erase(iter);
                }
            } else if ("Output" == op->type) {
                if (auto iter = check_output.find(op->name); iter != check_output.end()) {
                    check_output.erase(iter);
                }
            }
        }
        std::string unknown_inout;
        for (auto &tmp_name: check_input) {
            unknown_inout += (tmp_name + " ");
        }
        CHECK(unknown_inout.empty()) << "Unknown inputs: " << unknown_inout;
        for (auto &tmp_name: check_output) {
            unknown_inout += (tmp_name + " ");
        }
        CHECK(unknown_inout.empty()) << "Unknown outputs: " << unknown_inout;

        // 构建拓扑顺序
        TopoSortOperators();

        CHECK(to_po_operators_.size() == operators_.size()) << "Build wrong to_po queue";

        graph_state_ = GraphState::Complete;
        if (graph_ != nullptr) {
            graph_.reset();
            graph_ = nullptr;
        }
    }

    RuntimeGraph::GraphState RuntimeGraph::graph_state() const {
        return this->graph_state_;
    }

    const std::vector<std::shared_ptr<RuntimeOperator> > &
    RuntimeGraph::get_to_po_queues() const {
        return this->to_po_operators_;
    }

    const std::vector<std::shared_ptr<RuntimeOperator> > &RuntimeGraph::operators() const {
        return this->operators_;
    }

    void RuntimeGraph::ProbeNextLayer(const std::shared_ptr<RuntimeOperator> &current_op,
                                      const std::map<std::string, std::shared_ptr<RuntimeOperand>> &layer_output_operands) {
        // 当前节点的后继节点next_ops
        const auto &next_ops = current_op->output_operators;
        // 对所有后继节点进行遍历
        for (const auto &[next_rt_name, next_rt_operator]: next_ops) {
            // 后续节点的output
            const auto& layer_output_data = layer_output_operands.at(next_rt_name)->datas;
            // 得到后继节点的输入next_input_operands
            const auto &next_input_operands = next_rt_operator->input_operands;
            // 确定后继节点的输入来自于current_op
            if (next_input_operands.find(current_op->name) != next_input_operands.end()) {
                // 得到后继节点的关于current_op输出的输入空间 next_input_datas
                /**
                 * next_input_operands:
                 * {
                 *    输入1 -- current_op.name: current_op对应的输出空间
                 *    输入2 -- other_op.name: other_op对应的输出空间
                 * }
                 */
                std::vector<std::shared_ptr<ftensor> > &next_input_data = next_input_operands.at(current_op->name)->
                        datas;
                CHECK(next_input_data.size() == layer_output_data.size());
                // 将当前current_op的输出赋值到next_input_data中
                for (int i = 0; i < next_input_data.size(); ++i)
                    next_input_data.at(i) = layer_output_data.at(i);
            }
        }
    }

    std::map<std::string, std::vector<std::vector<std::shared_ptr<Tensor<float>>>>>
    RuntimeGraph::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float> > > > &inputs,
                          bool debug) {
        // 检查当前的执行图是否已经初始化完毕
        if (graph_state_ < GraphState::Complete)
            LOG(FATAL) << "Graph need be built!";
        CHECK(graph_state_ == GraphState::Complete)
             << "Graph status error, current state is " << int(graph_state_);

        CHECK(to_po_operators_.size() == operators_.size())
              << "Build wrong to po queues";

        // for (const auto &op: to_po_operators_)
        //     op->has_forward = false;

        // 赋值 input
        CHECK_EQ(inputs.size(), input_names_.size()) << "Build wrong number of inputs";
        for (int i = 0; i < inputs.size(); ++i) {
            auto &ipt_name = input_names_[i];
            // 进行读取operands
            for (const auto& output_operand: operators_maps_.at(ipt_name)->output_operands) {
                output_operand.second->datas = inputs.at(i);
            }
        }

        for (const auto &current_op: to_po_operators_) {
            if (current_op->type == "Input") {
                ProbeNextLayer(current_op, current_op->output_operands);
            } else if (current_op->type == "Output") {
                // TODO: This just make sure only one input in the input
                for (auto& output_operand: current_op->output_operands) {
                    output_operand.second = current_op->input_operands_seq.front();
                }
            } else {
                InferStatus status = current_op->layer->Forward();
                CHECK(status == InferStatus::bInferSuccess)
                     << current_op->layer->layer_name()
                     << " layer forward failed, error code: " << int(status);
                ProbeNextLayer(current_op, current_op->output_operands);
            }
        }

        std::map<std::string, std::vector<std::vector<std::shared_ptr<Tensor<float>>>>> final_outputs{};
        for (const auto &out_name: output_names_) {
            // 之前已经检查过 out_name 的合法性，这里不再检查
            final_outputs.insert({out_name, {}});
            const auto &output_op = operators_maps_.at(out_name);
            CHECK(!output_op->input_operands.empty()) << "Output from " << output_op->name << " is empty";
            const auto &output_operand = output_op->input_operands;
            for (const auto& item: output_operand) {
                final_outputs[out_name].emplace_back(item.second->datas);
            }
        }
        return final_outputs;
    }
}
