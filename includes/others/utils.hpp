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
}

#endif //BATMANINFER_UTILS_H
