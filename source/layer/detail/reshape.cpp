//
// Created by Mason on 2024/12/5.
//

#include <layer/detail/reshape.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
    InferStatus ReshapeLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the reshape layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        auto inputShape = inputs.at(0)->shapes();
        auto outputShape = outputs.at(0)->shapes();

        // TODO: 目前假设只有一个batch进行推理
        // 当allowzero设置为1时，形状数组中的零会被解释为将该维度的大小保持与输入数据的相同。
        // 当allowzero设置为0时，形状数组中的零会被解释为将该维度的大小设置为零，这通常是不允许的
        auto& output = outputs.at(0);

        auto size = output->shapes();

        auto temp = TensorClone(inputs.at(0));

        temp->Reshape(size, true);

        output = temp;

        // 判断allowzero

//        if (inputs.size() != outputs.size()) {
//            LOG(ERROR) << "The input and output tensor array size of the reshape "
//                       << "layer do not match";
//            return InferStatus::bInferFailedInputOutSizeMatchError;
//        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    ReshapeLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &reshape_layer) {
        CHECK(op != nullptr) << "The Reshape operator is empty";
        auto params = op->params;

        if (params.find("allowzero") == params.end()) {
            LOG(ERROR) << "Can not find the allow zero parameter";
            return ParseParameterAttrStatus::bParameterMissingAllowZero;
        }

        auto allow_zero_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("allowzero"));
        reshape_layer = std::make_shared<ReshapeLayer>(allow_zero_param->value);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    ReshapeLayer::ReshapeLayer(int allowzero): NonParamLayer("Reshape"), allowzero_(allowzero) {

    }

    LayerRegistererWrapper bReshapeLayerCreateInstance("Reshape", ReshapeLayer::CreateInstance);
}