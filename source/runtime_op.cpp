//
// Created by Mason on 2024/10/16.
//

#include <runtime/runtime_op.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    RuntimeOperator::~RuntimeOperator() {
        for (auto& [_, param] : this->params) {
            if (param != nullptr) {
                delete param;
                param = nullptr;
            }
        }
    }

    void RuntimeOperatorUtils::InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
        if (operators.empty()) {
            LOG(ERROR) << "Operators for init input shapes is empty!";
            return;
        }

        for (const auto &op: operators) {
            if (op->input_operands.empty())
                continue;
            else {
                // 获取当前操作符的输入操作数映射
                const std::map<std::string, std::shared_ptr<RuntimeOperand>> &input_operands_map = op->input_operands;
                // 初始化operator的输入控件
                for (const auto &[_, input_operand] : input_operands_map) {
                    // 遍历每个输入操作数。

                    const auto &type = input_operand->type;
                    CHECK(type == RuntimeDataType::kTypeFloat32)
                         << "The graph only support float32 yet!";
                    const auto &input_operand_shape = input_operand->shapes;
                    // 得到需要初始化空间
                    auto &input_datas = input_operand->datas;

                    // 确保输入操作数的形状不为空。
                    CHECK(!input_operand_shape.empty());
                    const int32_t batch = input_operand_shape.at(0);
                    // 获取批处理大小，并确保它是非负的（不支持动态批处理大小）。
                    CHECK(batch >= 0) << "Dynamic batch size is not supported!";
                    CHECK(input_operand_shape.size() == 2 ||
                          input_operand_shape.size() == 4 ||
                          input_operand_shape.size() == 3)
                          << "Unsupported tensor shape sizes: " << input_operand_shape.size();

                    if (!input_datas.empty())
                        CHECK_EQ(input_datas.size(), batch);
                    else
                        input_datas.resize(batch);
                }
            }
        }
    }

    void RuntimeOperatorUtils::InitOperatorOutput(const std::vector<ONNXOperator *> &onnx_operators,
                                                  const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
        CHECK(!onnx_operators.empty() && !operators.empty());
        CHECK(onnx_operators.size() == operators.size());
        for (uint32_t i = 0; i < onnx_operators.size(); ++i) {
            // 得到onnx原有的输出空间
            const std::vector<ONNXOperand *> operands = onnx_operators.at(i)->outputs;
            CHECK(operands.size() <= 1) << "Only support one node one output yet!";
            if (operands.empty())
                continue;
            CHECK(operands.size() == 1) << "Only support one output in the BatmanInfer";
            // 一个节点仅支持一个输出，实际在ONNX中一个节点拥有两个不同输出的情况也是不存在的
            ONNXOperand *operand = operands.front();
            const auto &runtime_op = operators.at(i);
            CHECK(operand != nullptr) << "Operand output is null";
            const std::vector<int32_t> &operand_shapes = operand->shape;
            // 得到需要初始化的输出空间
            const auto &output_tensors = runtime_op->output_operands;
            // 获取结点的输出张量应有形状
            const int32_t batch = operand_shapes.at(0);
            CHECK(batch >= 0) << "Dynamic batch size is not supported!";
            CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 4 || operand_shapes.size() == 3)
                  << "Unsupported shape sizes: " << operand_shapes.size();

            // 如果输出空间没有被初始化
            if (!output_tensors) {
                // 需要被初始化的输出张量
                std::shared_ptr<RuntimeOperand> output_operand = std::make_shared<RuntimeOperand>();
                // 将输出操作数赋变量
                output_operand->shapes = operand_shapes;
                output_operand->type = RuntimeDataType::kTypeFloat32;
                output_operand->name = operand->name + "_output";
                // 输出空间初始化
                for (int j = 0; j < batch; ++j) {
                    if (operand_shapes.size() == 4) {
                        sftensor output_tensor = TensorCreate(operand_shapes.at(1),
                                                              operand_shapes.at(2),
                                                              operand_shapes.at(3));
                        output_operand->datas.push_back(output_tensor);
                    } else if (operand_shapes.size() == 2) {
                        sftensor output_tensor = TensorCreate((uint32_t) operand_shapes.at(1));
                        output_operand->datas.push_back(output_tensor);
                    } else {
                        // current shape is 3
                        sftensor output_tensor = TensorCreate((uint32_t) operand_shapes.at(1),
                                                              (uint32_t) operand_shapes.at(2));
                        output_operand->datas.push_back(output_tensor);
                    }
                }
                runtime_op->output_operands = std::move(output_operand);
            } else {
                // If the input corner is not empty
                CHECK(batch == output_tensors->datas.size());
                CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
                CHECK(output_tensors->shapes == operand_shapes);
                // 逐批次检查输出空间是否合理，不合理则进行reshape
                for (uint32_t b = 0; b < batch; ++b) {
                    sftensor output_tensor = output_tensors->datas.at(b);
                    const std::vector<uint32_t> &tensor_shapes = output_tensor->shapes();
                    if (operand_shapes.size() == 4) {
                        if (tensor_shapes.at(0) != operand_shapes.at(1) ||
                            tensor_shapes.at(1) != operand_shapes.at(2) ||
                            tensor_shapes.at(2) != operand_shapes.at(3)) {
                            DLOG(WARNING)
                                    << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes = std::vector<uint32_t>{
                                    (uint32_t) operand_shapes.at(1), (uint32_t) operand_shapes.at(2),
                                    (uint32_t) operand_shapes.at(3)};
                            output_tensor->Reshape(target_shapes);
                        }
                    } else if (operand_shapes.size() == 2) {
                        if (tensor_shapes.at(0) != 1 ||
                            tensor_shapes.at(1) != operand_shapes.at(1) ||
                            tensor_shapes.at(2) != 1) {
                            DLOG(WARNING)
                                    << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes =
                                    std::vector<uint32_t>{(uint32_t) operand_shapes.at(1)};
                            output_tensor->Reshape(target_shapes);
                        }
                    } else {
                        // current shape is 3
                        if (tensor_shapes.at(0) != 1 ||
                            tensor_shapes.at(1) != operand_shapes.at(1) ||
                            tensor_shapes.at(2) != operand_shapes.at(2)) {
                            DLOG(WARNING)
                                    << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes = std::vector<uint32_t> {
                                    (uint32_t) operand_shapes.at(1),
                                    (uint32_t) operand_shapes.at(2)
                            };
                            output_tensor->Reshape(target_shapes);
                        }
                    }
                }
            }
        }
    }
}