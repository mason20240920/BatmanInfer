//
// Created by Mason on 2024/10/14.
//
// 在您的源文件中，在包含任何 ONNX 头文件之前，添加以下行：
#define ONNX_NAMESPACE onnx

#include <runtime/ir.h>
#include <onnx_conv/OnnxUtils.hpp>
#include <google/protobuf/util/json_util.h>
#include <onnx/onnx.pb.h>
#include <onnx/shape_inference/implementation.h>


#include <climits>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <utility>

namespace BatmanInfer {
    static bool type_is_integer(int type) {
        if (type == 1) return false;
        if (type == 2) return false;
        if (type == 3) return false;
        if (type == 4) return true;
        if (type == 5) return true;
        if (type == 6) return true;
        if (type == 7) return true;
        if (type == 8) return true;
        if (type == 9) return true;
        if (type == 10) return false;
        if (type == 11) return false;
        if (type == 12) return false;
        return false;
    }

    static const char *type_to_string(int type) {
        if (type == 1) return "f32";
        if (type == 2) return "f64";
        if (type == 3) return "f16";
        if (type == 4) return "i32";
        if (type == 5) return "i64";
        if (type == 6) return "i16";
        if (type == 7) return "i8";
        if (type == 8) return "u8";
        if (type == 9) return "bool";
        if (type == 10) return "cp64";
        if (type == 11) return "cp128";
        if (type == 12) return "cp32";
        return "null";
    }

    static const char *type_to_numpy_string(int type) {
        if (type == 1) return "float32";
        if (type == 2) return "float64";
        if (type == 3) return "float16";
        if (type == 4) return "int32";
        if (type == 5) return "int64";
        if (type == 6) return "int16";
        if (type == 7) return "int8";
        if (type == 8) return "uint8";
        if (type == 9) return "bool8";
        if (type == 10) return "csingle";
        if (type == 11) return "cdouble";
        if (type == 12) return "chalf";
        return "null";
    }

    static const char *type_to_dtype_string(int type) {
        if (type == 1) return "torch.float";
        if (type == 2) return "torch.double";
        if (type == 3) return "torch.half";
        if (type == 4) return "torch.int";
        if (type == 5) return "torch.long";
        if (type == 6) return "torch.short";
        if (type == 7) return "torch.int8";
        if (type == 8) return "torch.uint8";
        if (type == 9) return "torch.bool";
        if (type == 10) return "torch.complex64";
        if (type == 11) return "torch.complex128";
        if (type == 12) return "torch.complex32";
        return "null";
    }

    static size_t type_to_elemsize(int type) {
        if (type == 1) return 4;
        if (type == 2) return 8;
        if (type == 3) return 2;
        if (type == 4) return 4;
        if (type == 5) return 8;
        if (type == 6) return 2;
        if (type == 7) return 1;
        if (type == 8) return 1;
        if (type == 9) return 1;
        if (type == 10) return 8;
        if (type == 11) return 16;
        if (type == 12) return 4;
        return 0; // null
    }

    static int string_to_type(const char *s) {
        if (strcmp(s, "f32") == 0) return 1;
        if (strcmp(s, "f64") == 0) return 2;
        if (strcmp(s, "f16") == 0) return 3;
        if (strcmp(s, "i32") == 0) return 4;
        if (strcmp(s, "i64") == 0) return 5;
        if (strcmp(s, "i16") == 0) return 6;
        if (strcmp(s, "i8") == 0) return 7;
        if (strcmp(s, "u8") == 0) return 8;
        if (strcmp(s, "bool") == 0) return 9;
        if (strcmp(s, "cp64") == 0) return 10;
        if (strcmp(s, "cp128") == 0) return 11;
        if (strcmp(s, "cp32") == 0) return 12;
        return 0; // null
    }

    bool operator==(const ONNXParameter &lhs, const ONNXParameter &rhs) {
        if (lhs.type != rhs.type)
            return false;

        if (lhs.type == 0)
            return true;

        if (lhs.type == 1 && lhs.b == rhs.b)
            return true;

        if (lhs.type == 2 && lhs.i == rhs.i)
            return true;

        if (lhs.type == 3 && lhs.f == rhs.f)
            return true;

        if (lhs.type == 4 && lhs.s == rhs.s)
            return true;

        if (lhs.type == 5 && lhs.ai == rhs.ai)
            return true;

        if (lhs.type == 6 && lhs.af == rhs.af)
            return true;

        if (lhs.type == 7 && lhs.as == rhs.as)
            return true;

        return false;
    }

    ONNXAttribute::ONNXAttribute(const onnx::TensorProto &tensor) {
        // 初始化 Shape
        for (int i = 0; i < tensor.dims_size(); ++i)
            shape.emplace_back(tensor.dims(i));

        // 检查是否有原始值(现在只有float32类型的)
        if (tensor.has_raw_data()) {
            const std::string& raw_data = tensor.raw_data();
            data.resize(raw_data.size() / sizeof(float));
            std::memcpy(data.data(), raw_data.data(), raw_data.size());
        } else {
            // 使用 float_data 字段
            data.assign(tensor.float_data().begin(), tensor.float_data().end());
        }

        // 确定数据类型并初始化数据
        switch (tensor.data_type()) {
            case onnx::TensorProto::FLOAT:
                type = 1; // f32
                break;

            case onnx::TensorProto::DOUBLE:
                type = 2; // f64
                break;

            case onnx::TensorProto::INT32:
                type = 4; // i32
                break;

            case onnx::TensorProto::INT64:
                type = 5; // i64
                break;

                // 处理其他数据类型
            default:
//                // 处理 raw_data
//                if (!tensor.raw_data().empty()) {
//                    data.resize(tensor.raw_data().size());
//                    memcpy(data.data(), tensor.raw_data().data(), data.size());
//                    // 根据 raw_data 和 shape 设置正确的 type
//                    // 这里可以根据实际需求设置 type
//                }
                break;
        }
    }

    bool operator==(const ONNXAttribute &lhs, const ONNXAttribute &rhs) {
        if (lhs.type != rhs.type)
            return false;

        if (lhs.type == 0)
            return true;

        if (lhs.shape != rhs.shape)
            return false;

        if (lhs.data != rhs.data)
            return false;

        return true;
    }

    ONNXAttribute operator+(const ONNXAttribute &a, const ONNXAttribute &b) {
        ONNXAttribute c;

        if (a.type != b.type) {
            std::cerr << "concat attribute type mismatch\n";
            return c;
        }

        if (a.shape.size() != b.shape.size()) {
            std::cerr << "concat attribute shape rank mismatch\n";
            return c;
        }

        for (int i = 1; i < static_cast<int>(a.shape.size()); i++) {
            if (a.shape[i] != b.shape[i]) {
                std::cerr << "concat attribute shape mismatch\n";
                return c;
            }
        }

        c.type = a.type;
        c.shape = a.shape;
        c.shape[0] += b.shape[0]; // concat the first dim

        c.data.resize(a.data.size() + b.data.size());
        memcpy(c.data.data(), a.data.data(), a.data.size());
        memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());

        return c;
    }

    ONNXParameter ONNXParameter::parse_from_string(const std::string &value) {
        ONNXParameter p;
        p.type = 0;

        if (value == "None" || value == "()" || value == "[]") {
            return p;
        }

        if (value == "True" || value == "False") {
            // bool
            p.type = 1;
            p.b = value == "True";
            return p;
        }

        if (value[0] == '(' || value[0] == '[') {
            // list
            std::string lc = value.substr(1, value.size() - 2);
            std::istringstream lcss(lc);

            while (!lcss.eof()) {
                std::string elem;
                std::getline(lcss, elem, ',');

                if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) ||
                    (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                    // string
                    p.type = 7;
                    p.as.push_back(elem);
                } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                    // float
                    p.type = 6;
                    p.af.push_back(std::stof(elem));
                } else {
                    // integer
                    p.type = 5;
                    p.ai.push_back(std::stoi(elem));
                }
            }
            return p;
        }

        if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) ||
            (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
            // string
            p.type = 4;
            p.s = value;
            return p;
        }

        if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
            // float
            p.type = 3;
            p.f = std::stof(value);
            return p;
        }

        // integer
        p.type = 2;
        p.i = std::stoi(value);
        return p;
    }

    ONNXGraph::ONNXGraph() = default;

    ONNXGraph::~ONNXGraph() {
        for (auto x: operators)
            delete x;

    }

    ONNXGraph::ONNXGraph(const BatmanInfer::ONNXGraph &rhs) {}

    ONNXGraph &ONNXGraph::operator=(const BatmanInfer::ONNXGraph &rhs) {
        return *this;
    }

    /**
     * 是不是权重的输入
     * @param input_name
     * @param graph
     * @return
     */
    bool is_initializer(const std::string &input_name, const onnx::GraphProto &graph) {
        for (const auto &initializer: graph.initializer()) {
            if (initializer.name() == input_name) {
                return true; // Input is an initializer (weight)
            }
        }
        return false;
    }

    static void load_parameter(ONNXOperator *op,
                               const std::string &key,
                               const std::string &value) {
        op->params[key] = ONNXParameter::parse_from_string(value);
    }

    /**
     * 打印Tensor信息
     * @param tensor
     */
    static void print_tensor_info(const onnx::ValueInfoProto& tensor, ONNXOperand* operand)
    {
        operand->name = tensor.name();
        const auto& type = tensor.type().tensor_type();
        int data_type = type.elem_type();
        operand->type = map_onnx_type_to_custom_type(data_type);
        operand->shape.clear();
        for (const auto& dim : type.shape().dim()) {
            if (dim.has_dim_value())
                operand->shape.push_back(dim.dim_value());
            else
                operand->shape.push_back(-1);
        }
    }

    void find_input_tensor_info(const std::string& input_name,
                                const onnx::GraphProto& graph,
                                ONNXOperator *op,
                                size_t index)
    {

        for (const auto& value_info : graph.value_info()) {
            if (value_info.name() == input_name) {
                op->input_names[index] = input_name;
                print_tensor_info(value_info, op->inputs[index]);
                return;
            }
        }

        // 如果在节点中找不到，检查图的输入
        for (const auto& input_tensor : graph.input()) {
            if (input_tensor.name() == input_name) {
                op->input_names[index] = input_name;
                print_tensor_info(input_tensor, op->inputs[index]);
                return;
            }
        }

        for (const auto& output_tensor : graph.output()) {
            if (output_tensor.name() == input_name) {
                op->input_names[index] = input_name;
                print_tensor_info(output_tensor, op->inputs[index]);
                return;
            }
        }
    }


    static void load_input_key(ONNXOperator *op,
                               const onnx::NodeProto& node,
                               const onnx::GraphProto& graph) {
        op->input_names.resize(op->inputs.size());
        for (size_t i = 0; i < op->inputs.size(); i++) {
            // 查找不是权重的值
            const ONNXOperand* operand = op->inputs[i];
            if (!is_initializer(operand->name, graph)) {
                find_input_tensor_info(operand->name, graph, op, i);
            }
        }
    }

    static void load_output_key(ONNXOperator *op,
                                const onnx::GraphProto& graph) {
        // 获取最后一个算子的输出作为输入
        const onnx::ValueInfoProto& output_info = graph.output(0);
        const std::string& operand_name = output_info.name();

//        // 创建一个新的操作数对象
//        ONNXOperand *r = new_operand(operand_name);
//        r->consumer.push_back(op);
//        op->inputs.emplace_back(r);

        // 确保输出类型是 Tensor
        const onnx::TypeProto& output_type = output_info.type();
        if (!output_type.has_tensor_type()) {
            std::cerr << "Output is not a tensor type." << std::endl;
            return;
        }

        // 获取ONNX Tensor类型
        const onnx::TypeProto::Tensor& tensor_type = output_type.tensor_type();
        int onnx_data_type = tensor_type.elem_type();

        // 将ONNX类型映射为自定义类型
        int custom_type = map_onnx_type_to_custom_type(onnx_data_type);

        // 获取输出的Shape
        const onnx::TensorShapeProto& shape = tensor_type.shape();
        std::vector<int32_t> output_shape;
        for (int j = 0; j < shape.dim_size(); ++j) {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
            if (dim.has_dim_value()) {
                output_shape.push_back(dim.dim_value());
            } else {
                output_shape.push_back(-1);  // 未定义的维度
            }
        }
        op->input_names.resize(1);
        op->inputs[0]->name = "output";
        op->inputs[0]->type = custom_type;
        op->inputs[0]->shape = output_shape;
        op->input_names[0] = "output";
    }

    /**
     * 从Node结点里获取ONNXAttribute
     * @param node
     * @param attribute_name
     * @return
     */
    ONNXAttribute get_attribute_from_node(const onnx::TensorProto &tensor) {
        ONNXAttribute custom_attr;

        custom_attr = ONNXAttribute(tensor);
        return custom_attr;
    }

    /**
     * 查看是否是权重
     * @param input_name 输入的参数名
     * @param graph
     * @return
     */
    const onnx::TensorProto *is_initializer_tensor(const std::string &input_name, const onnx::GraphProto &graph) {
        for (const auto &initializer: graph.initializer()) {
            if (initializer.name() == input_name) {
                return &initializer; // Input is an initializer (weight)
            }
        }
        return nullptr;
    }



    /**
     * Function to get the weight names from a node
     * @param node 节点信息
     * @return
     */
    std::map<std::string, const onnx::TensorProto *> get_weight_names_from_node(const onnx::NodeProto &node,
                                                                                const onnx::GraphProto &graph) {
        std::map<std::string, const onnx::TensorProto *> weight_names;

        // Iterate through the node's inputs
        for (int i = 0; i < node.input_size(); ++i) {
            const std::string &input_name = node.input(i);

            // Check if this input is part of the initializer (i.e., it's a weight)
            auto tensor = is_initializer_tensor(input_name, graph);
            if (tensor) {
                weight_names[input_name] = tensor;
            }
        }

        for (int i = 0; i < node.attribute_size(); ++i) {
            const auto attribute = node.attribute(i);
            const std::string &attribute_name = attribute.name();
            std::cout << "Hello World" << std::endl;
        }

        return weight_names;
    }

    /**
     * 解析配置文件或参数设置
     * @param op 自定义操作符
     * @param node 操作符结点
     * @param graph 图的结构
     */
    void load_attribute(ONNXOperator *op,
                        const onnx::NodeProto &node,
                        const onnx::GraphProto &graph) {
        auto attribute_names = get_weight_names_from_node(node, graph);
        for (const auto &attr_info: attribute_names) {
            ONNXAttribute a = get_attribute_from_node(*attr_info.second);
            op->attrs[attr_info.first] = a;
        }
    }

    int ONNXGraph::load(const std::string &model_path) {

        // 读取ONNX模型文件
        onnx::ModelProto modelProto;

        // read ONNX Model
        bool success = onnx_read_proto_from_binary(model_path.c_str(),
                                                   reinterpret_cast<google::protobuf::Message *>(&modelProto));

        // 读取 ONNX 模型文件
        if (!success) {
            fprintf(stderr, "Failed to read ONNX model from %s\n", model_path.c_str());
            return -1;
        }

        // 先执行模型推理，执行形状推理，才能获取中间算子的输入和输出
        onnx::shape_inference::InferShapes(modelProto);

        // 对模型结构进行修改(增加一个input作为ONNXOperator的开始节点
        // 获取模型的输入
        auto graph_info = modelProto.graph();


        // 读取操作符和操作数的数量
        int operator_count = 0;
        int operand_count = 0;

        getOperatorAndOperandCount(modelProto,
                                   operator_count,
                                   operand_count);

        // 新增两个operator, 一个是input,一个是output
        operator_count += 2;

        // 获取算子图
//        auto graph = modelProto.graph();

        // 获取算子的信息
        for (int i = 0; i < operator_count; ++i) {
            // 对于第一个算子，进行手动新增，这是一个input node节点
            if (i == 0) {
                const std::string& type = "Input";
                const std::string& name = "Input";
                int input_count = 0;
                int output_count = 1;

                ONNXOperator *op = new_operator(type, name);

                // 新增一个输出的结点
                std::string operand_name;

                ONNXOperand *r = new_operand(operand_name);
                r->producer = op;
                op->outputs.emplace_back(r);

                // 获取第一个输入
                const onnx::ValueInfoProto& input_info = graph_info.input(0);
                // 获取类型信息
                const onnx::TypeProto& input_type = input_info.type();

                // 确保输入类型是 Tensor
                if (!input_type.has_tensor_type()) {
                    std::cerr << "Input is not a tensor type." << std::endl;
                    return -1;
                }

                // 获取ONNX Tensor类型
                const onnx::TypeProto::Tensor& tensor_type = input_type.tensor_type();
                int onnx_data_type = tensor_type.elem_type();

                // 将ONNX类型映射为自定义类型
                int custom_type = map_onnx_type_to_custom_type(onnx_data_type);

                // 获取输入的Shape
                const onnx::TensorShapeProto& shape = tensor_type.shape();
                std::vector<int32_t> input_shape;
                for (int j = 0; j < shape.dim_size(); ++j) {
                    const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
                    if (dim.has_dim_value()) {
                        input_shape.push_back(dim.dim_value());
                    } else {
                        input_shape.push_back(-1);  // 未定义的维度
                    }
                }
                r->producer = op;
                r->name = "input";
                r->type = custom_type;
                r->shape = input_shape;

                // 跳过后面的输入
                continue;
            } else if (i == operator_count - 1) {
                const std::string& type = "Output";
                const std::string& name = "Output";
                int input_count = 1;
                int output_count = 0;

                // 创建Output算子
                ONNXOperator *op = new_operator(type, name);

                for (int j = 0; j < input_count; j++) {
                    // 获取第 j 个输入的名称
                    const std::string &operand_name = operators.at(operator_count - 2)->outputs[0]->name;

                    // 获取输入的操作数
                    ONNXOperand *r = get_operand(operand_name);
                    r->consumers.push_back(op);
                    // 输入的消费者
                    op->inputs.push_back(r);

                }

                load_output_key(op, graph_info);

                break;
            }


            // 获取每一个算子
            const onnx::NodeProto &node = graph_info.node(i - 1);

            const std::string& type = node.op_type();
            const std::string& name = node.name();
            int input_count = get_data_input_count(node);
            int output_count = node.output_size();

            ONNXOperator *op = new_operator(type, name);

            for (int j = 0; j < input_count; j++) {
                // 获取第 j 个输入的名称
                const std::string &operand_name = node.input(j);

                // 获取输入的操作数
                ONNXOperand *r = get_operand(operand_name);
                r->consumers.push_back(op);
                // 输入的消费者
                op->inputs.push_back(r);
            }

            for (int j = 0; j < output_count; j++) {
                const std::string& operand_name = node.output(j);

                ONNXOperand *r = new_operand(operand_name);
                r->producer = op;
                op->outputs.emplace_back(r);
            }

            // 对操作符进行权重参数加载
            load_attribute(op, node, graph_info);

            // 对操作数进行权重参数加载
            load_input_key(op, node, graph_info);
        }

        return 0;

    }

    ONNXOperand *ONNXGraph::get_operand(const std::string &name) {
        for (ONNXOperand *r: operands) {
            if (r->name == name)
                return r;
        }
        return nullptr;
    }

    ONNXOperator *ONNXGraph::new_operator(const std::string &type, const std::string &name) {
        auto *op = new ONNXOperator;
        op->type = type;
        op->name = name;
        operators.emplace_back(op);
        return op;
    }

    ONNXOperand *ONNXGraph::new_operand(const std::string &name) {
        auto *r = new ONNXOperand;
        r->name = name;
        operands.emplace_back(r);
        return r;
    }

    ONNXOperator *
    ONNXGraph::new_operator_before(const std::string &type, const std::string &name, const ONNXOperator *cur) {
        auto *op = new ONNXOperator;
        op->type = type;
        op->name = name;
        operators.insert(std::find(operators.begin(), operators.end(), cur), op);
        return op;
    }

    ONNXOperator *
    ONNXGraph::new_operator_after(const std::string &type, const std::string &name, const ONNXOperator *cur) {
        auto *op = new ONNXOperator;
        op->type = type;
        op->name = name;
        operators.insert(std::find(operators.begin(), operators.end(), cur) + 1, op);
        return op;
    }

    const ONNXOperand *ONNXGraph::get_operand(const std::string &name) const {
        for (const ONNXOperand *r : operands)
        {
            if (r->name == name)
                return r;
        }
        return nullptr;
    }

}