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
            for (const auto[_, output_value]: outputs) {
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
            auto& output = iter->second->data();

            Halide::Buffer<float> input(output);

            auto shapes = iter->second->shapes();

            auto dims = input.dimensions();

            std::vector<Halide::Var> vars(dims);
            for (int i = 0; i < dims; ++i)
                vars[i] = Halide::Var("v" + std::to_string(i));

            // 创建 Halide 函数
            Halide::Func f_exp, f_softmax;

            // 计算指数值
            f_exp(to_expr_vector(vars)) = Halide::exp(input(to_expr_vector(vars)));

            // 定义制定维度的归约操作
            int dim_size = shapes[softmax_dim_];
            Halide::RDom r(0, dim_size); // 在指定维度范围进行归约

            // 创建动态维度的 Buffer
            std::vector<halide_dimension_t> halide_dims(shapes.size());
            for (int i = 0; i < shapes.size(); i++) {
                halide_dims[i] = {0, static_cast<int32_t>(shapes[i]), 1}; // {min, extent, stride}
            }
            Halide::Buffer<float> exp_buffer(nullptr, static_cast<int>(shapes.size()), halide_dims.data());

            // 将 f_exp 的输出写入 exp_buffer
            f_exp.realize(exp_buffer);

            // 使用 CBLAS 进行归约求和
            std::vector<float> sum_buffer(exp_buffer.number_of_elements() / dim_size, 0.0f);

            int outer_size = exp_buffer.number_of_elements() / dim_size; // 归约维度以外的元素数量
#pragma omp parallel for
            for (int i = 0; i < outer_size; i++) {
                int offset = i * dim_size;
                sum_buffer[i] = cblas_sasum(dim_size, exp_buffer.data() + offset, 1);
            }

            // 将归一化结果写回输入
            f_softmax(to_expr_vector(vars)) = f_exp(to_expr_vector(vars)) / sum_buffer[vars[softmax_dim_]];

            // 实现输出
            f_softmax.realize(input);


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