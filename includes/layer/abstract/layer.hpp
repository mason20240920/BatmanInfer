//
// Created by Mason on 2024/10/28.
//

#ifndef BATMAN_INFER_LAYER_HPP
#define BATMAN_INFER_LAYER_HPP
#include <string>
#include "status_code.hpp"
#include <data/Tensor.hpp>
#include <runtime/runtime_op.hpp>

namespace BatmanInfer {
    class RuntimeOperator;
    class Layer {
    public:
        explicit Layer(std::string layer_name):layer_name_(std::move(layer_name)) {}

        virtual ~Layer() = default;

        /**
         * Layer的执行函数
         * @param inputs inputs 层的输入
         * @param outputs outputs 层的输出
         * @return 执行的状态
         */
        virtual InferStatus Forward(
                const std::map<std::string , std::shared_ptr<Tensor<float>>>& inputs,
                std::map<std::string, std::shared_ptr<Tensor<float>>>& outputs);

        /**
         * Layer的执行函数
         * @return 执行的状态
         */
        virtual InferStatus Forward();

        /**
         * 返回层的权重
         * @return  返回的权重
         */
        virtual const std::shared_ptr<Tensor<float>>& weights() const;

        /**
         * 返回层的偏移量
         * @return  返回层的偏移量
         */
        virtual const std::shared_ptr<Tensor<float>>& bias() const;

        /**
         * 设置Layer的权重
         * @param weights  权重
         */
        virtual void set_weights(const std::shared_ptr<Tensor<float>>& weights);

        /**
    * 设置Layer的偏移量
    * @param bias 偏移量
    */
        virtual void set_bias(
                const std::shared_ptr<Tensor<float>>& bias);

        /**
         * 设置Layer的权重
         * @param weights 权重
         */
        virtual void set_weights(const std::vector<float>& weights);

        /**
         * 设置Layer的偏移量
         * @param bias 偏移量
         */
        virtual void set_bias(const std::vector<float>& bias);

        /**
         * 返回层的名称
         * @return 层的名称
         */
        virtual const std::string& layer_name() const { return this->layer_name_;}

        /**
         * 设置层的执行算子
         * @param runtime_operator 该层的执行算子
         */
        void set_runtime_operator(
                const std::shared_ptr<RuntimeOperator>& runtime_operator);

    protected:
        std::weak_ptr<RuntimeOperator> runtime_operator_;
        std::string layer_name_;  // Layer的名称
    };
}

#endif //BATMAN_INFER_LAYER_HPP
