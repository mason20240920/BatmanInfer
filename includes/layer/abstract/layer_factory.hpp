//
// Created by Mason on 2024/10/29.
//

#ifndef BATMAN_INFER_LAYER_FACTORY_HPP
#define BATMAN_INFER_LAYER_FACTORY_HPP
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include <runtime/runtime_op.hpp>

namespace BatmanInfer {
    class LayerRegister {
    public:
        /**
         * 类型别名 Creator: 函数指针类型
         * 返回类型: ParseParameterAttrStatus
         * 接受两个参数:
         * i. `const std::shared_ptr<RuntimeOperator> &op`: 一个指向`RuntimeOperator`对象的常量智能指针的引用
         * ii. `std::shared_ptr<Layer> &layer`: 一个指向`Layer`对象的智能指针的引用
         */
        typedef ParseParameterAttrStatus (*Creator)(
                const std::shared_ptr<RuntimeOperator> &op,
                std::shared_ptr<Layer> &layer);

        /**
         * 类型别名: `CreateRegistry`: 是一个`std::map`
         * `std::string`: 键的类型
         * `Creator`: 值的类型: 指向特定函数类型的参数指针
         */
        typedef std::map<std::string, Creator> CreateRegistry;
    public:
        /**
         * 向注册表注册算子
         * @param layer_type  算子的类型
         * @param creator 需要注册算子的注册表
         */
        static void RegisterCreator(const std::string &layer_type,
                                    const Creator &creator);

        /**
         * 通过算子参数op来初始化Layer
         * @param op 保存了初始化Layer信息的算子
         * @return  初始化的Layer
         */
        static std::shared_ptr<Layer> CreateLayer(
                const std::shared_ptr<RuntimeOperator> &op);

        /**
         * 返回算子的注册表
         * @return 算子的注册表
         */
        static CreateRegistry &Registry();

        /**
         * 返回所有已经被注册算子的类型
         * @return  注册算子的类型列表
         */
        static std::vector<std::string> layer_types();
    };

    class LayerRegistererWrapper {
    public:
        LayerRegistererWrapper(const std::string &layer_type,
                               const LayerRegister::Creator &creator) {
            LayerRegister::RegisterCreator(layer_type, creator);
        }
    };
}

#endif //BATMAN_INFER_LAYER_FACTORY_HPP
