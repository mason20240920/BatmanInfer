//
// Created by Mason on 2024/10/13.
//

#include <onnx_conv/OnnxUtils.hpp>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <algorithm>

namespace BatmanInfer {
    bool onnx_read_proto_from_binary(const char* filepath, google::protobuf::Message* message) {
        std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            fprintf(stderr, "open failed %s\n", filepath);
            return false;
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
        codedstr.SetTotalBytesLimit(INT_MAX);
#else
        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX/2);
#endif

        bool success = message->ParseFromCodedStream(&codedstr);

        fs.close();

        return success;
    }
    bool onnx_write_proto_from_binary(const char* filepath, const google::protobuf::Message* message) {
        std::ofstream fs(filepath);
        if (fs.fail()) {
            fprintf(stderr, "open failed %s\n", filepath);
            return false;
        }
        message->SerializeToOstream(&fs);
        fs.close();
        return true;
    }

    void get_operator_and_operand_count(const onnx::ModelProto& model,
                                        int& operator_count,
                                        int& operand_count) {
        const onnx::GraphProto& graph = model.graph();

        // 获取节点数量
        operator_count = graph.node_size();

        // 获取初始化张量以及输入输出的数量
        int initializer_count = graph.initializer_size();
        int input_count = graph.input_size();
        int output_count = graph.output_size();

        // 计算操作数的总数
        operand_count = initializer_count + input_count + output_count;
    }

    // 将 ONNX 数据类型映射到自定义整数类型
    int map_onnx_type_to_custom_type(int onnx_type) {
        switch (onnx_type) {
            case onnx::TensorProto::UNDEFINED:
                return 0; // null
            case onnx::TensorProto::FLOAT:
                return 1; // f32
            case onnx::TensorProto::DOUBLE:
                return 2; // f64
            case onnx::TensorProto::FLOAT16:
                return 3; // f16
            case onnx::TensorProto::INT32:
                return 4; // i32
            case onnx::TensorProto::INT64:
                return 5; // i64
            case onnx::TensorProto::INT16:
                return 6; // i16
            case onnx::TensorProto::INT8:
                return 7; // i8
            case onnx::TensorProto::UINT8:
                return 8; // u8
            case onnx::TensorProto::BOOL:
                return 9; // bool
            case onnx::TensorProto::COMPLEX64:
                return 10; // cp64
            case onnx::TensorProto::COMPLEX128:
                return 11; // cp128
            case onnx::TensorProto::BFLOAT16:
                return 12; // cp32
            default:
                return -1; // 未知类型
        }
    }

    bool is_data_input(const std::string& input_name) {
        return input_name.find("weight") == std::string::npos &&
               input_name.find("bias") == std::string::npos &&
               input_name.find("onnx");
    }

    bool is_constant_value(const std::string& text) {
        const std::string& pattern = "Constant";
        // 创建 Boyer-Moore 搜索器
        return std::search(
                text.begin(), text.end(),
                pattern.begin(), pattern.end()
        ) != text.end();
    }

    /**
     * 函数用于判断输入是否为数据输入
     * @param input_name
     * @return
     */
    int get_data_input_count(const onnx::NodeProto &node) {
        int data_input_count = 0;
        for (int i = 0; i < node.input_size(); ++i) {
            const std::string& input_name = node.input(i);
            if (is_data_input(input_name)) {
                ++data_input_count;
            }
        }
        return data_input_count;
    }

    void convertRepeatedPtrFieldToVector(const google::protobuf::RepeatedPtrField<std::string>& repeatedPtrField, std::vector<std::string>& af) {
        // Clear the vector to ensure it's empty before copying
        af.clear();

        // Copy elements from RepeatedPtrField to std::vector
        af.reserve(repeatedPtrField.size()); // Reserve space to improve performance
        for (const auto& item : repeatedPtrField) {
            af.push_back(item); // Copy each string
        }
    }

}