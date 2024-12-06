//
// Created by Mason on 2024/11/27.
//


#include <layer/detail/constant.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {

    ConstantLayer::ConstantLayer(std::vector<float> value):value_(std::move(value)), NonParamLayer("Constant") {

    }

    InferStatus ConstantLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                       std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        // 目前输出的是一个[batch size, 一个是constant出来的张量
        const uint32_t batch_size = 1;

        if (outputs.empty()) {
            LOG(ERROR) << "The output of tensor array of the constant layer is empty";
            return InferStatus::bInferFailedOutputEmpty;
        }

        for (uint32_t i = 0; i < batch_size; ++i) {
            auto& output = outputs.at(i);
            output = std::make_shared<ftensor>(value_.size());
            output->Fill(value_, true);
        }
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus ConstantLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                          std::shared_ptr<Layer> &constant_layer) {
        CHECK(op != nullptr) << "Constant operator is nullptr";
        const auto& params = op->params;

        if (params.find("value") == params.end()) {
            LOG(ERROR) << "Can not find the value parameter";
            return ParseParameterAttrStatus::bParameterMissingValue;
        }

        auto value = params.at("value");

        std::vector<float> tmp_float_vec;

        if (auto int_param = dynamic_cast<RuntimeParameterIntArray*>(value.get())) {
            tmp_float_vec = convert_to_int_vector(int_param);
        } else if (auto float_param = dynamic_cast<RuntimeParameterFloat*>(value.get())) {
            tmp_float_vec = std::vector<float>{float_param->value};
        }

        constant_layer = std::make_shared<ConstantLayer>(tmp_float_vec);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bConstantCreateInstance("Constant", ConstantLayer::CreateInstance);

}