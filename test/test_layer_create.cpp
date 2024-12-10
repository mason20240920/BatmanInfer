//
// Created by Mason on 2024/10/30.
//
#include <layer/abstract/layer_factory.hpp>
#include <gtest/gtest.h>
#include <map>
#include <Halide.h>
using namespace BatmanInfer;

static LayerRegister::CreateRegistry *RegistryGlobal() {
    static auto *kRegistry = new LayerRegister::CreateRegistry();
    CHECK(kRegistry != nullptr) << "Global layer register init failed!";
    return kRegistry;
}

TEST(test_registry, registry1) {
    using namespace BatmanInfer;
    LayerRegister::CreateRegistry  *registry1 = RegistryGlobal();
    LayerRegister::CreateRegistry  *registry2 = RegistryGlobal();

    LayerRegister::CreateRegistry *registry3 = RegistryGlobal();
    LayerRegister::CreateRegistry *registry4 = RegistryGlobal();
    auto *a = new float{3};
    auto *b = new float{4};
    ASSERT_EQ(registry1, registry2);
}

ParseParameterAttrStatus MyTestCreator(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &layer) {

    layer = std::make_shared<Layer>("test_layer");
    return ParseParameterAttrStatus::bParameterAttrParseSuccess;
}

TEST(test_registry, registry2) {
    using namespace BatmanInfer;
    LayerRegister::CreateRegistry registry1 = LayerRegister::Registry();
    LayerRegister::CreateRegistry registry2 = LayerRegister::Registry();
    ASSERT_EQ(registry1, registry2);
    LayerRegister::RegisterCreator("test_type", MyTestCreator);
    LayerRegister::CreateRegistry registry3 = LayerRegister::Registry();
    ASSERT_EQ(registry3.size(), 3);
    ASSERT_NE(registry3.find("test_type"), registry3.end());
}

TEST(test_registry, create_layer) {
    LayerRegister::RegisterCreator("test_type_1", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "test_type_1";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_util) {
    LayerRegistererWrapper kReluGetInstance("test_type_2", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "test_type_2";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_relu_forward) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.ReLU";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(3, 4, 3);
    input_tensor->Rand();
}

TEST(test_registry, create_layer_softmax_forward) {
    // 1. 初始化一个运行时算子
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "Relu";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>( 1, 8, 8);
    input_tensor->Rand();
    input_tensor->Show();
    std::map<std::string, sftensor> input_map{
            {"input", input_tensor}
    };

    sftensor output = std::make_shared<ftensor>(1, 8, 8);
    std::map<std::string, sftensor> output_map {
            {"output", output},
    };

    layer->Forward(input_map, output_map);

//    output->Show();

}

TEST(test_registry, test_halide_buffer) {
    try {
        // 创建 halide_buffer_t
        halide_buffer_t h_data_ = {0};
        size_t size = 1024; // 假设一维数据
        h_data_.dimensions = 2;
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];
        h_data_.dim[0] = {0, 8, 1, 0}; // 一维数据
        h_data_.dim[1] = {0, 128, 8, 0};
        h_data_.host = new uint8_t[size * sizeof(float)];
        h_data_.type = halide_type_t(halide_type_float, 32, 1); // float 类型

        // 获取维度信息
        int dimensions = h_data_.dimensions;

        // 定义 Halide 的变量和函数
        std::vector<Halide::Var> vars(dimensions);
        for (int i = 0; i < dimensions; i++) {
            vars[i] = Halide::Var("dim" + std::to_string(i));
        }

        Halide::Func random_fill("random_fill");

        // 使用 Halide 内置随机数生成器
        random_fill(vars) = Halide::random_float();

        // 调度：并行化和向量化
        if (dimensions >= 2) {
            random_fill.parallel(vars[0]).vectorize(vars[1], 8);
        } else if (dimensions == 1) {
            random_fill.vectorize(vars[0], 8);
        }

        // 将生成的随机数写入 halide_buffer_t
        Halide::Buffer<float> output(h_data_);
        random_fill.realize(output);

        // 打印第一个元素
        float* data = reinterpret_cast<float*>(h_data_.host);
        std::cout << "First element: " << data[0] << std::endl;

        // 释放内存
        delete[] h_data_.host;
        delete[] h_data_.dim;
    } catch (const Halide::Error &e) {
        std::cerr << "Halide error: " << e.what() << std::endl;
    }
}
