//
// Created by Mason on 2024/11/5.
//

#include <layer/detail/sigmoid.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <Halide.h>

namespace BatmanInfer {
    InferStatus SigmoidLayer::Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                                      std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the relu layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }
        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                       << "not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        auto output_iter = outputs.begin();
        for (const auto&[_, input_tensor]: inputs) {
            // 将 halide_buffer_t 转换为Halide::Buffer
            Halide::Buffer<float> input(input_tensor->data());
            Halide::Buffer<float> output(output_iter->second->data());

            // 确定维度数
            int dimensions = input.dimensions();

            // 定义 Halide 变量
            std::vector<Halide::Var> vars(dimensions);
            for (int i = 0; i < dimensions; ++i)
                vars[i] = Halide::Var("dim" + std::to_string(i));

            // 定义 Halide 函数
            Halide::Func sigmoid;
            // 构建动态索引的表达式
            std::vector<Halide::Expr> indices;
            indices.reserve(dimensions);
            for (int i = 0; i < dimensions; ++i) {
                indices.push_back(vars[i]);
            }

            Halide::Expr val = input(indices);
            sigmoid(indices) = 1.0f / (1.0f + exp(-val));

            // 调度策略: 对最外层维度进行并行化
            if (dimensions > 0)
                sigmoid.parallel(vars[0]);

            sigmoid.realize(output);

            // 结果同步回 halide_buffer_t
            output.copy_to_host();

            ++output_iter;
        }
        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus SigmoidLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &sigmoid_layer) {
        CHECK(op != nullptr) << "Sigmoid operator is nullptr";
        sigmoid_layer = std::make_shared<SigmoidLayer>();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    // 使用工具类注册算子
    LayerRegistererWrapper bSigmoidGetInstance("Sigmoid", SigmoidLayer::GetInstance);
}