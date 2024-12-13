//
// Created by Mason on 2024/11/4.
//

#include <layer/detail/softmax.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>
#include <Halide.h>
#include <cblas.h>

namespace BatmanInfer {
    // Helper function: 将 std::vector<Var> 转换为 Halide 的参数展开
    template <typename T>
    std::vector<Halide::Expr> to_expr_vector(const std::vector<T> &vars) {
        std::vector<Halide::Expr> exprs(vars.begin(), vars.end());
        return exprs;
    }

    InferStatus SoftmaxLayer::Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                                      std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the softmax layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }
        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the softmax layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        for (const auto&[_, input_value]: inputs) {
            for (const auto&[_, output_value]: outputs) {
                if (input_value->shapes() != output_value->shapes()) {
                    LOG(ERROR) << "The input and output tensor shapes in softmax mismatch";
                    return InferStatus::bInferFailedInputOutSizeMatchError;
                }
                CHECK(input_value->shapes() == output_value->shapes()) <<
                "The input and output tensor shapes in softmax mismatch";
            }
        }

        auto iter = outputs.begin();

        // 获取输入的维度数量
        for (const auto&[_, input_value] : inputs) {
            iter->second = TensorClone(input_value);
            auto& output_halide = iter->second->data();

            // 将输入和输出包装为 Halide::Buffer
            Halide::Buffer<float> input(input_value->data());
            Halide::Buffer<float> output(output_halide);

            // 确定确定的维度
            const int axis = softmax_dim_;

            // 检查输入和输出缓冲区是否定义
            CHECK(input.defined() && output.defined()) << "Buffer not properly defined";

            // 定义 Halide 变量
            Halide::Var x("x"), y("y"), z("z"), t("t");

            Halide::Func max_val("max_val"), exp_values("exp_values"), sum_exp("sum_exp"), softmax("softmax");

            if (input.dimensions() == 1) {
                // Handle 1D case
                Halide::RDom r(0, input.width(), "r");
                max_val() = maximum(input(r));
                exp_values(x) = exp(input(x) - max_val());
                sum_exp() = sum(exp_values(r));
                softmax(x) = exp_values(x) / sum_exp();
                softmax.parallel(x).vectorize(x, 8);
            } else if (input.dimensions() == 2) {
                // Handle 2D case with axis
                if (axis == 0) { // Column-wise
                    Halide::RDom r(0, input.height(), "r");
                    max_val(x) = maximum(input(x, r));
                    exp_values(x, y) = exp(input(x, y) - max_val(x));
                    sum_exp(x) = sum(exp_values(x, r));
                    softmax(x, y) = exp_values(x, y) / sum_exp(x);
                    softmax.parallel(x).vectorize(y, 8);
                } else {
                    Halide::RDom r(0, input.width(), "r");
                    // 注意: 这里y是固定的,
                    max_val(y) = maximum(input(r, y));
                    exp_values(x, y) = exp(input(x, y) - max_val(y));
                    sum_exp(y) = sum(exp_values(r, y));
                    softmax(x, y) = exp_values(x, y) / sum_exp(y);
                    // OpenMP and SIMD
                    softmax.parallel(y).vectorize(x, 8);
                }
            } else if (input.dimensions() == 3) {
                // Handle 3D case with axis
                if (axis == 0) { // Depth-wise (z-axis)
                    Halide::RDom r(0,
                           input.dim(2).extent(),
                           "r");
                    max_val(x, y) = maximum(input(x, y, r));
                    exp_values(x, y, z) = exp(input(x, y, z) - max_val(x, y));
                    sum_exp(x, y) = sum(exp_values(x, y, r));
                    softmax(x, y, z) = exp_values(x, y, z) / sum_exp(x, y);
//                    softmax.parallel(y).vectorize(x, 8);
                } else if (axis == 1) {
                    Halide::RDom r(0, input.dim(1).extent(), "r");
                    max_val(x, z) = maximum(input(x, r, z));
                    exp_values(x, y, z) = exp(input(x, y, z) - max_val(x, z));
                    sum_exp(x, z) = sum(exp_values(x, r, z));
                    softmax(x, y, z) = exp_values(x, y, z) / sum_exp(x, z);
                    softmax.parallel(z).vectorize(x, 8);
                } else {
                    Halide::RDom r(0, input.dim(0).extent(), "r");
                    max_val(y, z) = maximum(input(r, y, z));
                    exp_values(x, y, z) = exp(input(x, y, z) - max_val(y, z));
                    sum_exp(y, z) = sum(exp_values(r, y, z));
                    softmax(x, y, z) = exp_values(x, y, z) / sum_exp(y, z);
                    softmax.parallel(x).vectorize(y, 8);
                }
            } else if (input.dimensions() == 4) {
                if (axis == 0) {
                    Halide::RDom r(0,
                                   input.dim(3).extent(),
                                   "r");
                    max_val(x, y, z) = maximum(input(x, y, z, r));
                    exp_values(x, y, z, t) = exp(input(x, y, z, t) - max_val(x, y, z));
                    sum_exp(x, y, z) = sum(exp_values(x, y, z, r));
                    softmax(x, y, z, t) = exp_values(x, y, z, t) / sum_exp(x, y, z);
                } else if (axis == 1) {
                    Halide::RDom r(0,
                                   input.dim(2).extent(),
                                   "r");
                    max_val(x, y, t) = maximum(input(x, y, r, t));
                    exp_values(x, y, z, t) = exp(input(x, y, z, t) - max_val(x, y, t));
                    sum_exp(x, y, t) = sum(exp_values(x, y, r, t));
                    softmax(x, y, z, t) = exp_values(x, y, z, t) / sum_exp(x, y, t);
                } else if (axis == 2) {
                    Halide::RDom r(0,
                                   input.dim(1).extent(),
                                   "r");
                    max_val(x, z, t) = maximum(input(x, r, z, t));
                    exp_values(x, y, z, t) = exp(input(x, y, z, t) - max_val(x, z, t));
                    sum_exp(x, z, t) = sum(exp_values(x, r, z, t));
                    softmax(x, y, z, t) = exp_values(x, y, z, t) / sum_exp(x, z, t);
                } else if (axis == 3) {
                    Halide::RDom r(0,
                                   input.dim(0).extent(),
                                   "r");
                    max_val(y, z, t) = maximum(input(r, y, z, t));
                    exp_values(x, y, z, t) = exp(input(x, y, z, t) - max_val(y, z, t));
                    sum_exp(y, z, t) = sum(exp_values(r, y, z, t));
                    softmax(x, y, z, t) = exp_values(x, y, z, t) / sum_exp(y, z, t);
                }
            }

            softmax.realize(output);
            ++iter;
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus
    SoftmaxLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                              std::shared_ptr<Layer> &softmax_layer) {
        CHECK(op != nullptr) << "Softmax operator is nullptr";
        const std::map <std::string, std::shared_ptr<RuntimeParameter>> &params = op->params;

        if (params.find("axis") == params.end()) {
            LOG(ERROR) << "Can not find the axis parameter";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        auto axis_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("axis"));
        if (axis_param == nullptr) {
            LOG(ERROR) << "Can not find the axis parameter";
            return ParseParameterAttrStatus::bParameterMissingAxis;
        }

        softmax_layer = std::make_shared<SoftmaxLayer>(axis_param->value);
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    SoftmaxLayer::SoftmaxLayer(int dim) : NonParamLayer("Softmax"), softmax_dim_(dim){

    }

    // 使用工具类注册算子
    LayerRegistererWrapper bSoftmaxGetInstance("Softmax", SoftmaxLayer::GetInstance);


}