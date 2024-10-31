//
// Created by Mason on 2024/10/30.
//
#include <layer/abstract/layer_factory.hpp>
#include <gtest/gtest.h>

static BatmanInfer::LayerRegister::CreateRegistry *RegistryGlobal() {
    static auto *kRegistry = new BatmanInfer::LayerRegister::CreateRegistry();
    CHECK(kRegistry != nullptr) << "Global layer register init failed!";
    return kRegistry;
}

TEST(test_registry, registry1) {
    using namespace BatmanInfer;
    LayerRegister::CreateRegistry  *registry1 = RegistryGlobal();
    LayerRegister::CreateRegistry  *registry2 = RegistryGlobal();
    ASSERT_EQ(registry1, registry2);
}

using namespace BatmanInfer;
ParseParameterAttrStatus MyTestCreator(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &layer) {
    layer = std::make_shared<Layer>("test_layer");
    return ParseParameterAttrStatus::bParameterAttrParseSuccess;
}

