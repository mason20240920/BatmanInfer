//
// Created by Mason on 2024/11/1.
//
#include <gtest/gtest.h>
#include <layer/abstract/layer_factory.hpp>
using namespace BatmanInfer;

TEST(test_registry, create_layer_find) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.Sigmoid";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    // 评价注册是否成功
    ASSERT_NE(layer, nullptr);
}