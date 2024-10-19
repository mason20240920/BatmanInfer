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
    using namespace BatmanInfer;
    const std::string& model_path = "./model/model.onnx";
    RuntimeGraph graph(model_path);
    const bool init_success = graph.Init();
    // 如果这里加载失败，请首先考虑
    ASSERT_EQ(init_success, true);
    graph.Build("input", "output");
}