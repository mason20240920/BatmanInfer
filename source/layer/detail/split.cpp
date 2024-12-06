//
// Created by Mason on 2024/12/6.
//

#include <layer/detail/split.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <others/utils.hpp>

namespace BatmanInfer {
    SplitLayer::SplitLayer(const int axis): NonParamLayer("Split"), axis_(axis) {

    }

    InferStatus SplitLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the split layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        // 1. 两个输入, 1个是真的input, 2个是Constant
        // 2. 先取constant的值
        const auto& constant_input = inputs.at(1);
        // 3. 获取切分的结果
        const auto const_val = constant_input->values(true);
        const auto const_int_val = convert_to_uint32(const_val);
        // 4. 获取结果
        const auto input = inputs.at(0);

        // TODO: 5. 确定轴方向，目前只用粗暴的axis_ - 1
        auto real_axis = axis_ - 1;

        const auto& output_size = outputs.size();
        CHECK(output_size == const_val.size()) << "The number of split layer is wrong";


        outputs = input->Split(real_axis, const_int_val);

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    SplitLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                               std::shared_ptr<Layer> &split_layer) {
        CHECK(op != nullptr) << "Split Operator is nullptr";

        auto params = op->params;

        if (params.find("axis") == params.end()) {
            LOG(ERROR) << "Split layer not found axis parameters";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        auto axis_ = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("axis"));

        if (axis_ == nullptr) {
            LOG(ERROR) << "Split layer not found axis parameters";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        split_layer = std::make_shared<SplitLayer>(axis_->value);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bSplitCreateInstance("Split", SplitLayer::CreateInstance);
}