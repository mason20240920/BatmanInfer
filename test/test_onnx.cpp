//
// Created by Mason on 2024/10/13.
//

#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>
#include <onnx/onnx_pb.h>
#include <google/protobuf/text_format.h>
#include <others/utils.hpp>
#include <runtime/ir.h>

TEST(test_ir, onnx_graph_ops) {
    const std::string& modelPath = "../model_files/operators/concat_model_single_input.onnx";

    // 初始化Protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // 读取ONNX模型文件
    onnx::ModelProto modelProto;
    std::ifstream fin(modelPath, std::ios::in | std::ios::binary);
    if (!fin) {
        LOG(ERROR) <<  "Cannot open file: " << modelPath;
        return;
    }

    if (!modelProto.ParseFromIstream(&fin)) {
        LOG(ERROR) <<  "Failed to parse ONNX model.";
        return;
    }

    fin.close();

    // 获取计算图 (Graph)
    const onnx::GraphProto& graph = modelProto.graph();

    // 获取算子列表 (节点列表)
    const auto& nodes = graph.node();

    // 遍历算子列表，打印每个算子的名称
    for (int i = 0; i < nodes.size(); ++i) {
        const onnx::NodeProto& node = nodes.Get(i);
        // 打印算子名称
        LOG(INFO) << "Operator " << i << ": " << node.op_type();
    }

    // 清理 protobuf
    google::protobuf::ShutdownProtobufLibrary();
}



TEST(test_ir, onnx_graph_operands_and_params) {
    using namespace BatmanInfer;
    const std::string& modelPath = "./model/model.onnx";

    // 初始化Protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // 读取ONNX模型文件
    onnx::ModelProto modelProto;
    std::ifstream fin(modelPath, std::ios::in | std::ios::binary);
    if (!fin) {
        LOG(ERROR) <<  "Cannot open file: " << modelPath;
        return;
    }

    if (!modelProto.ParseFromIstream(&fin)) {
        LOG(ERROR) <<  "Failed to parse ONNX model.";
        return;
    }

    fin.close();

    // 获取计算图 (Graph)
    const onnx::GraphProto& graph = modelProto.graph();

    // 获取算子列表 (节点列表)
    const auto& nodes = graph.node();

    // 获取初始化器列表 (权重参数)
    std::unordered_map<std::string, onnx::TensorProto> initializerMap;
    for (const auto& initializer : graph.initializer())
        initializerMap[initializer.name()] = initializer;

    // 遍历算子列表，打印每个算子的名称
    for (int i = 0; i < nodes.size(); ++i) {
        const onnx::NodeProto& node = nodes.Get(i);
        // 打印算子名称
        LOG(INFO) << "Operator " << i << ": " << node.op_type();

        // 检查节点的输入
        for (const auto& input : node.input()) {
            if (initializerMap.find(input) != initializerMap.end()) {
                const onnx::TensorProto& weight = initializerMap[input];
                LOG(INFO) << " Weight parameters: " << input ;
            }
        }
    }
}

TEST(test_ir, onnx_model_load) {
    using namespace BatmanInfer;

    const std::string& modelPath = "./model/model.onnx";

    // 加载图地址
    auto graph = new ONNXGraph();
    graph->load(modelPath);

    // 获取算子的结果
    std::cout << "ONNX Graph build successfully" << std::endl;
}