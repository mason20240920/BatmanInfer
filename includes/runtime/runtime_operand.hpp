//
// Created by Mason on 2024/10/15.
//

#ifndef BATMAN_INFER_RUNTIME_OPERAND_HPP
#define BATMAN_INFER_RUNTIME_OPERAND_HPP
#include <vector>
#include <string>
#include <memory>
#include "status_code.hpp"
#include "runtime_datatype.hpp"
#include <data/tensor.hpp>

namespace BatmanInfer {
    /**
     * 计算节点输入输出的操作数
     */
    struct RuntimeOperand {
        // 操作数的名称
        std::string name;
        // 操作数的形状
        std::vector<int32_t> shapes;
        // 存储操作数
        std::shared_ptr<Tensor<float>> data;
        // 操作数的类型，一般是float
        RuntimeDataType type = RuntimeDataType::kTypeUnknown;
    };
}

#endif //BATMAN_INFER_RUNTIME_OPERAND_HPP
