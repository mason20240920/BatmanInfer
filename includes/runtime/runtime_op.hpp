//
// Created by Mason on 2024/10/15.
//

#ifndef BATMAN_INFER_RUNTIME_OP_HPP
#define BATMAN_INFER_RUNTIME_OP_HPP

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace BatmanInfer {
    class Layer;

    /**
     * 计算图中的计算节点
     */
    struct RuntimeOperator {
        virtual ~RuntimeOperator();

        // 计算节点的名称
        std::string name;
        // 计算节点的类型
        std::string type;
        // 节点对应的计算Layer
        std::shared_ptr<Layer> layer;
        // 节点的输出节点名称
        std::vector<std::string> output_names;
        // 节点的输出操作数
        std::map<std::string, std::shared_ptr<RuntimeOperand>> output_operands;

        // 节点的输入操作数
        std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
        // 节点的输入操作数，顺序排列
        std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;
        // 输出节点的名字和节点对应
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;

        // 算子的参数信息
        std::map<std::string, std::shared_ptr<RuntimeParameter>> params;
        // 算子的属性信息, 内涵权重信息
        std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
    };

    class RuntimeOperatorUtils {
    public:
        /**
         * 如果图是第一次运行，则根据节点输入operand的形状准备后续Layer计算中所需要的Tensor
         * 如果图是第二次以上，则检查输入operand形状和operand中张量的形状是否匹配
         * @param operators 计算图中的计算节点
         */
        static void InitOperatorInput(
                const std::vector<std::shared_ptr<RuntimeOperator>>& operators);


        /**
         * 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
         * 如果图是第二次以上运行，则检查输出operand的形状和operand中张量的形状是否匹配
         * @param onnx_operators
         * @param operators
         */
        static void InitOperatorOutput(
                const std::vector<ONNXOperator *> &onnx_operators,
                const std::vector<std::shared_ptr<RuntimeOperator>> &operators);
    };
}

#endif //BATMAN_INFER_RUNTIME_OP_HPP
