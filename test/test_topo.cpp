//
// Created by Mason on 2024/10/19.
//

#include <runtime/ir.h>
#include <runtime/runtime_ir.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

TEST(test_ir_topo, topo) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/resetnet18_batch1.onnx";
    RuntimeGraph graph(model_path);
    const bool init_success = graph.Init();
    // 如果这里加载失败，请首先考虑
    ASSERT_EQ(init_success, true);
    graph.Build("input", "output");
    const auto &to_po_queues = graph.get_to_po_queues();

    int index = 0;
    for (const auto &operator_ : to_po_queues) {
        LOG(INFO) << "Index: " << index << " Type: " << operator_->type
                  << " Name: " << operator_->name;
        index += 1;
    }
}

TEST(test_ir_topo, build_output_ops) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/simple_square_model.onnx";
    RuntimeGraph graph(model_path);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    graph.Build("input", "output");
    const auto &to_po_queues = graph.get_to_po_queues();

    int index = 0;
    for (const auto& operator_: to_po_queues) {
        LOG(INFO) << "Index: " << index << " Name: " << operator_->name;
        index+=1;
    }
}

TEST(test_ir_topo, build_output_ops2) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/simple_square_model.onnx";
    RuntimeGraph graph(model_path);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    graph.Build("input", "output");
    const auto &to_po_queues = graph.get_to_po_queues();

    int index = 0;
    for (const auto &operator_ : to_po_queues) {
        LOG(INFO) << "Operator name: " << operator_->name;
        for (const auto &pair : operator_->output_operators)
            LOG(INFO) << "Output: " << pair.first;
        LOG(INFO) << "-----------------------";
        index += 1;
    }
}

TEST(test_ir_topo, build1_status) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/simple_square_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build("input", "output");
    ASSERT_EQ(int(graph.graph_state()), 0);
}

TEST(test_ir_topo, build1_output_tensors) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/simple_square_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build("input", "output");
    ASSERT_EQ(int(graph.graph_state()), 0);

    const auto &ops = graph.operators();
    for (const auto &op: ops) {
        LOG(INFO) << op->name;
        // 打印op输出空间的张量
        const auto &operand = op->output_operands;
        if (!operand || operand->datas.empty())
            continue;
        const uint32_t batch_size = operand->datas.size();
        LOG(INFO) << "batch: " << batch_size;

        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto &data = operand->datas.at(i);
            LOG(INFO) << "channel: " << data->channels()
                      << " height: " << data->rows() << " cols: " << data->cols();
        }
    }
}

TEST(test_ir_attribute, test_attr) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/resnet18.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build("input", "output");
    ASSERT_EQ(int(graph.graph_state()), 0);

    const auto &ops = graph.operators();
    for (const auto &op: ops) {
        // 获取最大池化的算子
        auto op_type = op->type;
        if (op_type == "MaxPool") {
            auto parameters = op->params;
            for (auto parameter: parameters) {
                LOG(INFO) << parameter.first;
            }
        }
    }
}

TEST(test_ir_attribute, test_ops) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/simple_conv_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build("input", "output");
    ASSERT_EQ(int(graph.graph_state()), 0);

    // 输出数据
    std::vector<std::shared_ptr<Tensor<float>>> data_lst;
//    graph.ProbeNextLayer(graph.operators().at(0), data_lst);
    std::cout << "Runtime execute successfully" << std::endl;
}
