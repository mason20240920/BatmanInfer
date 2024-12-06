//
// Created by Mason on 2024/10/13.
//

#ifndef BATMANINFER_UTILS_H
#define BATMANINFER_UTILS_H
#include <string>
#include <onnx/onnx_pb.h>
#include <runtime/runtime_datatype.hpp>

namespace BatmanInfer {
    std::string ShapeStr(const std::vector<int> &shapes);

    /**
    * 从onnx的type类型转为Runtime类型
    */
    RuntimeDataType convert_runtime_data(const int onnx_type);

    /**
     * @brief 从float数组转为uint32_t数组
     * @param input
     * @return
     */
    std::vector<uint32_t> convert_to_uint32(const std::vector<float>& input);
}

#endif //BATMANINFER_UTILS_H
