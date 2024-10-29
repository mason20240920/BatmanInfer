//
// Created by Mason on 2024/10/29.
//

#include <layer/abstract/layer_factory.hpp>
#include <runtime/runtime_ir.hpp>

namespace BatmanInfer {
    void LayerRegister::RegisterCreator(const std::string &layer_type,
                                        const Creator &creator) {
        CHECK(creator != nullptr);
        CreateRegistry& registry = Registry();
        CHECK_EQ(registry.count(layer_type), 0)
                << "Layer type: " << layer_type << " has already registered!";
        registry.insert({layer_type, creator});
    }

    LayerRegister::CreateRegistry& LayerRegister::Registry() {
        static auto* kRegistry = new CreateRegistry();
        CHECK(kRegistry != nullptr) << "Global layer register init failed!";
        return *kRegistry;
    }

    std::shared_ptr<Layer> LayerRegister::CreateLayer(const std::shared_ptr<RuntimeOperator> &op) {
        CreateRegistry& registry = Registry();
        const std::string& layer_type = op->type;
        LOG_IF(FATAL, registry.count(layer_type) <= 0)
              << "Can not find the layer type: " << layer_type;
        const auto& creator = registry.find(layer_type)->second;

        LOG_IF(FATAL, !creator) << "Layer creator is empty!";
        std::shared_ptr<Layer> layer; // 空的Layer
        const auto& status = creator(op, layer);
        LOG_IF(FATAL, status != ParseParameterAttrStatus::bParameterAttrParseSuccess)
              << "Create the layer: " << layer_type
              << " failed, error code: " << int(status);
        return layer;
    }

    std::vector<std::string> LayerRegister::layer_types() {
        std::vector<std::string> layer_types;
        static CreateRegistry& registry = Registry();
        for (const auto& [layer_type, _] : registry)
            layer_types.push_back(layer_type);
        return layer_types;
    }
}