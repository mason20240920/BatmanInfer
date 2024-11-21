//
// Created by Mason on 2024/10/16.
//

#ifndef BATMAN_INFER_RUNTIME_IR_HPP
#define BATMAN_INFER_RUNTIME_IR_HPP

#include "ir.h"
#include <runtime/runtime_operand.hpp>
#include "runtime_op.hpp"
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace BatmanInfer {
    /**
     * 计算图结构, 由多个计算节点和节点之间的数据流程图组成
     */
    class RuntimeGraph {
    public:
        /**
         * 初始化计算图
         * @param model_path 模型的路径
         */
        RuntimeGraph(std::string  model_path);

        /**
         * 设置模型路径
         * @param model_path 设置模型路径
         */
        void set_model_path(const std::string& model_path);

        /**
         * 返回模型路径
         * @return
         */
        const std::string &model_path();

        /**
         * 计算图的初始化
         * @return 是否初始化成功
         */
        bool Init();

        /**
         * 获取操作符
         * @return
         */
        const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

        /**
         * 构建计算图
         * @param input_names_strings 计算图输入节点的名称
         * @param output_names_strings 计算图输出节点的名称
         */
        void Build(const std::vector<std::string> &input_names_strings,
                   const std::vector<std::string> &output_names_strings);

        const std::vector<std::shared_ptr<RuntimeOperator>> &
        get_to_po_queues() const;

        /**
         * 按照顺序依次执行算子序列 (to_po_operators) 中每个算子的 Forward 方法即可
         * @param inputs 顺序必须按照初始化模型时传入的 input_names_ 的顺序
         * @param debug
         * @return
         */
        std::vector< std::vector< std::shared_ptr< Tensor<float> > > > Forward(
            const std::vector< std::vector< std::shared_ptr<Tensor<float>> > > &inputs,
            bool debug);


    private:
        /**
         * 初始化Batman Infer计算图节点中的输入操作数
         * @param inputs ONNX中的输入操作数
         * @param runtime_operator 计算图节点
         */
        static void InitGraphOperatorsInput(
                const std::vector<ONNXOperand *> & inputs,
                const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 当前节点的所有后继节点进行依次的遍历，并将当前节点 == 输出 == 赋值 给后继节点
         * @param current_op 表示当前的算子
         * @param layer_output_data 表示当前算子的输出: 在这个函数中，我们需要把当前算子的输出赋值到它后继节点的输入中
         */
        static void ProbeNextLayer(const std::shared_ptr<RuntimeOperator> &current_op,
                            const std::vector<std::shared_ptr<Tensor<float>>> &layer_output_data);


        /**
         * 初始化
         * @param outputs
         * @param runtime_operator
         */
        static void InitGraphOperatorsOutput(
                const std::vector<ONNXOperand *> &outputs,
                const std::shared_ptr<RuntimeOperator> &runtime_operator);


        /**
         * 初始化Batman Infer 计算图中的节点属性
         * @param attrs ONNX节点属性
         * @param runtime_operator  计算图节点
         */
        static void
        InitGraphAttrs(const std::map<std::string, ONNXAttribute> &attrs,
                       const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 初始化Batman Infer计算图的结点参数
         * @param params ONNX中的参数属性
         * @param runtime_operator 计算图节点
         */
        static void
        InitGraphParams(const std::map<std::string, ONNXParameter> &params,
                        const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 拓扑排序制作推理图
         */
        void TopoSortOperators();

        std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);


    private:
        enum class GraphState {
            NeedInit = -2,
            NeedBuild = -1,
            Complete = 0,
        };

    public:
        /**
         * 返回模型当前的状态
         * @return 返回模型当前的状态
         */
        GraphState graph_state() const;
    private:
        GraphState graph_state_ = GraphState::NeedInit;
        /**
         * 计算图输入节点的名称
         */
        std::vector<std::string> input_names_;

        /**
         * 计算图输出节点的名称
         */
        std::vector<std::string> output_names_;

        /**
         * 模型的文件路径
         */
        std::string model_path_;

        /**
         * 算子
         */
        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;
        /**
         * 拓扑的算子
         */
        std::vector<std::shared_ptr<RuntimeOperator>> to_po_operators_;

        /**
         * Onnx的graph
         */
        std::unique_ptr<ONNXGraph> graph_;
    };
}

#endif //BATMAN_INFER_RUNTIME_IR_HPP
