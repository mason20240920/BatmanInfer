//
// Created by Mason on 2024/10/14.
//
// 在您的源文件中，在包含任何 ONNX 头文件之前，添加以下行：
#define ONNX_NAMESPACE onnx

#include <runtime/ir.h>
#include <onnx_conv/OnnxUtils.hpp>
#include <google/protobuf/util/json_util.h>
//#include <onnx/onnx.pb.h>
#include <onnx/shape_inference/implementation.h>


#include <cstdint>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <string>
#include <stack>
#include <others/utils.hpp>

namespace BatmanInfer {
    static bool type_is_integer(const int type) {
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

    static const char *type_to_string(const int type) {
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

    static const char *type_to_numpy_string(const int type) {
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

    static const char *type_to_dtype_string(const int type) {
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

    static size_t type_to_elemsize(const int type) {
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

        // 确定数据类型并初始化数据
        switch (tensor.data_type()) {
            case onnx::TensorProto::FLOAT: {
                type = 1; // f32
                // 检查是否有原始值(现在只有float32类型的)
                if (tensor.has_raw_data()) {
                    const std::string& raw_data = tensor.raw_data();
                    data.resize(raw_data.size() / sizeof(float));
                    std::memcpy(data.data(), raw_data.data(), raw_data.size());
                } else {
                    // 使用 float_data 字段
                    data.assign(tensor.float_data().begin(), tensor.float_data().end());
                }
                break;
            }

            case onnx::TensorProto::DOUBLE:
                type = 2; // f64
                break;

            case onnx::TensorProto::INT32:
                type = 4; // i32
                break;

            case onnx::TensorProto::INT64: {
                type = 5; // i64
                // 检查是否有原始值(现在只有float32类型的)
                if (tensor.has_raw_data()) {
                    const std::string& raw_data = tensor.raw_data();
                    std::vector<int64_t> tmp_data;
                    tmp_data.resize(raw_data.size() / sizeof(int64_t));
                    std::memcpy(tmp_data.data(), raw_data.data(), raw_data.size());
                    data = convert_int64_to_float(tmp_data);
                } else {
                    // 使用 float_data 字段
                    data.assign(tensor.float_data().begin(), tensor.float_data().end());
                }
                break;
            }

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
        return std::any_of(
                graph.initializer().begin(),
                graph.initializer().end(),
                [&input_name](const onnx::TensorProto& initializer) {
                    return initializer.name() == input_name;
                }
        );
    }

//    void parse_tensor_attribute

    /**
     * 新增加载Parameters的函数
     * @param op
     * @param node
     */
    static void load_parameter(ONNXOperator *op,
                               const onnx::NodeProto& node) {

        for (const auto& attribute: node.attribute()) {
            const auto& parameter_name = attribute.name();
            ONNXParameter parameter;
            switch (attribute.type()) {
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_UNDEFINED:
                    parameter = ONNXParameter();
                    break;
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT:
                    parameter = ONNXParameter(attribute.f());
                    break;
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
                    parameter = ONNXParameter(attribute.i());
                    break;
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING:
                    parameter = ONNXParameter(attribute.s());
                    break;
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR: {
                    // TODO: Implement the function to implements Tensor
                    auto ret = attribute.tensors();
                    parameter = ONNXParameter(attribute.t());
                    break;
                }
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH:
                    // TODO: Implement the function to implements Graph
                    std::cout << "GRAPH" << std::endl;
                    std::cout << "  Graph name: " << attribute.g().name() << std::endl;
                    break;
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS: {
                    std::vector<float> tmpFloats;
                    convertRepeatedFieldToVector<float, float>(attribute.floats(), tmpFloats);
                    parameter = ONNXParameter(tmpFloats);
                    break;
                }
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS: {
                    std::vector<int> tmpInts;
                    convertRepeatedFieldToVector<int64_t, int>(attribute.ints(), tmpInts);
                    parameter = ONNXParameter(tmpInts);
                    break;
                }
                case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS: {
                    std::vector<std::string> tmpParams;
                    convertRepeatedPtrFieldToVector(attribute.strings(), tmpParams);
                    parameter = ONNXParameter(tmpParams);
                    break;
                }
                default:
                    std::cout << "Unknown type" << std::endl;
                    break;
            }
            op->params[parameter_name] = parameter;
        }
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
        while (operand->shape.size() < 3) {
            operand->shape.insert(operand->shape.begin(), 1);
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
            // 是不是权重参数
            const auto is_init = is_initializer(operand->name, graph);
            // 是不是Constant算子参数
            const auto is_constant = is_constant_value(operand->name);
            if (!is_init && !is_constant) {
                find_input_tensor_info(operand->name, graph, op, i);
            }
        }
    }

    static void load_output_key(ONNXOperator *op,
                                const onnx::GraphProto& graph,
                                const int index,
                                const std::string& name) {
        // 获取最后一个算子的输出作为输入
        const onnx::ValueInfoProto& output_info = graph.output(index);
        const std::string& operand_name = output_info.name();

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

        while (output_shape.size() < 3) {
            output_shape.insert(output_shape.begin(), 1);
        }
        op->input_names.resize(1);
        op->inputs[0]->name = name;
        op->inputs[0]->type = custom_type;
        op->inputs[0]->shape = output_shape;
        op->input_names[0] = name;
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

    /**
     * 根据这个获取input节点被多少个Operators使用
     * @param graph
     * @param input_name
     * @return
     */
    int count_input_usage(const onnx::GraphProto &graph,
                          const std::string& input_name) {
        int usage_count = 0;
        for (const auto& node: graph.node()) {
            for (const auto& input: node.input()) {
                if (input_name == input)
                    ++usage_count;
            }
        }
        return usage_count;
    }

    /**
     * 根据这个获取output节点多少个Operators使用
     * @param graph
     * @param output_name
     * @return
     */
    int count_output_usage(const onnx::GraphProto &graph,
                           const std::string& output_name) {
        int usage_count = 0;
        for (const auto& node : graph.node()) {
            for (const auto& output: node.output()) {
                if (output_name == output)
                    ++usage_count;
            }
        }
        return usage_count;
    }


    ONNXOperand* ONNXGraph::append_constant_operand(const std::string &name,
                                                    const onnx::GraphProto &graph,
                                                    const onnx::NodeProto& node) {
        const onnx::ValueInfoProto* value_info = nullptr;
        auto find_value_info = [&name](const auto& container) -> const onnx::ValueInfoProto* {
            for (const auto& info : container) {
                if (info.name() == name) {
                    return &info;
                }
            }
            return nullptr;
        };
        value_info = find_value_info(graph.value_info());
        if (!value_info) value_info = find_value_info(graph.output());
        if (!value_info) value_info = find_value_info(graph.input());
        if (!value_info) {
            throw std::runtime_error("Output not found: " + name);
        }
        // 获取类型信息
        const auto& type_proto = value_info->type();
        if (!type_proto.has_tensor_type()) {
            throw std::runtime_error("Output is not a tensor type: " + name);
        }

        // 获取ONNX Tensor类型
        const onnx::TypeProto::Tensor& tensor_type = type_proto.tensor_type();
        int onnx_data_type = tensor_type.elem_type();

        // 将ONNX类型映射为自定义类型
        int custom_type = map_onnx_type_to_custom_type(onnx_data_type);
        // 获取形状信息
        std::vector<int32_t> input_shape;
        if (tensor_type.has_shape()) {
            const auto& shape = tensor_type.shape();
            input_shape.reserve(shape.dim_size());  // 预分配内存
            for (const auto& dim : shape.dim()) {
                input_shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
            }
        }

        // 如果是标量就加一个维度
        if (input_shape.empty())
            // 目前是一个batch size，一个标量
            input_shape = {1, 1};

        if (input_shape.size() == 1)
            input_shape = {1, input_shape[0]};

        auto *r = new ONNXOperand;
        r->name = name;
        r->type = custom_type;
        r->shape = input_shape;
//        r->data_ = get_constant_value(node);
        operands.emplace_back(r);
        return r;
    }

    /**
     * 获取onnx模型的输入的inputs
     * @param graph
     * @return
     */
    std::vector<std::string_view> load_extra_nodes(const onnx::GraphProto& graph,
                                                   bool is_input) {
        std::vector<std::string_view> input_names;
        if (is_input) {
            // 预分配内存
            input_names.reserve(graph.input_size());
            for (const auto& input : graph.input()) {
                input_names.emplace_back(input.name());
            }
            return input_names;
        }
        input_names.reserve(graph.output_size());
        for (const auto& output: graph.output())
            input_names.emplace_back(output.name());
        return input_names;
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

        get_operator_and_operand_count(modelProto,
                                       operator_count,
                                       operand_count);

        auto input_names = load_extra_nodes(graph_info, true);
        auto output_names = load_extra_nodes(graph_info, false);

        auto input_size = static_cast<int>(input_names.size());
        auto output_size = static_cast<int>(output_names.size());

        // 新增两个operator, 一个是input,一个是output
        operator_count += input_size;
        operator_count += output_size;

        // 获取算子的信息
        for (int i = 0; i < operator_count; ++i) {
            // 对于前面的input的算子
            if (i < input_size) {
                const std::string type("Input");
                const std::string name(input_names.at(i));

                ONNXOperator *op = new_operator(type, name);

                // 新增一个输出的结点
                std::string operand_name;

                // 获取出度
                int output_count = count_input_usage(graph_info, name);

                // 标记错误
                int error_flag = 0;

#pragma omp parallel for
                for (int op_i = 0; op_i < output_count; ++op_i) {
                    // 如果error报错，直接跳过后面
                    if (error_flag == 1) continue;
                    ONNXOperand *r = new_operand(operand_name);
                    r->producer = op;
                    op->outputs.emplace_back(r);

                    // 获取第一个输入
                    const onnx::ValueInfoProto& input_info = graph_info.input(i);
                    // 获取类型信息
                    const onnx::TypeProto& input_type = input_info.type();

                    // 确保输入类型是 Tensor
                    if (!input_type.has_tensor_type()) {
#pragma omp critical
                        error_flag = 1;
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
                    // 如果shapes不满4个维度，怎么办?
                    while (input_shape.size() < 3) {
                        input_shape.insert(input_shape.begin(), 1);
                    }
                    r->producer = op;
                    r->name = name;
                    r->type = custom_type;
                    r->shape = input_shape;
                }

                if (error_flag == 1)
                    return -1;

                // 跳过后面的输入
                continue;
            } else if (i >= operator_count - output_size) {
                const std::string type("Output");
                const std::string name (output_names[i - operator_count + output_size]);
                int output_count = count_output_usage(graph_info, name);

                // 创建Output算子
                ONNXOperator *op = new_operator(type, name);

                for (int j = 0; j < output_count; j++) {
                    // 获取输入的操作数
                    ONNXOperand *r = get_operand(name);
                    r->consumers.push_back(op);
                    // 输入的消费者
                    op->inputs.push_back(r);

                }

                load_output_key(op,
                                graph_info,
                                i - operator_count + output_size,
                                name);

                continue;
            }


            // 获取每一个算子
            const onnx::NodeProto &node = graph_info.node(i - input_size);

            const std::string& type = node.op_type();
            const std::string& name = node.name();
            int input_count = get_data_input_count(node);
            int output_count = node.output_size();

            ONNXOperator *op = new_operator(type, name);

            for (int j = 0; j < input_count; j++) {
                // 获取第 j 个输入的名称
                const std::string &operand_name = node.input(j);
                // 判断是不是initializer的input
                if (is_initializer(operand_name, graph_info)) {
                    continue;
                }
                // 获取输入的操作数
                ONNXOperand *r = get_operand(operand_name);
                r->consumers.push_back(op);
                // 输入的消费者
                op->inputs.push_back(r);
            }

            for (int j = 0; j < output_count; j++) {
                // 判断是不是Constant算子
                const std::string& operand_name = node.output(j);
                ONNXOperand *r = nullptr;
                if (type == "Constant")
                    r = append_constant_operand(operand_name,
                                            graph_info,
                                            node);
                else
                    r = new_operand(operand_name);
                r->producer = op;
                op->outputs.emplace_back(r);
            }

            // 对操作符进行权重参数加载
            load_attribute(op, node, graph_info);

            // 对操作数进行权重参数加载
            load_input_key(op, node, graph_info);

            // 对操作数的参数进行加载
            load_parameter(op, node);
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