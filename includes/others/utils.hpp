//
// Created by Mason on 2024/10/13.
//

#ifndef BATMANINFER_UTILS_H
#define BATMANINFER_UTILS_H
#include <string>
#include <onnx/onnx_pb.h>
#include <runtime/runtime_datatype.hpp>
#include "runtime/runtime_attr.hpp"
#include <map>

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

    /**
     * @brief 根据key的substring获取值
     * @param attributes: 原始的属性
     * @param substring: key的substring
     * @return
     */
    std::shared_ptr<RuntimeAttribute> find_keys_with_substring(const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attributes,
                                                               const std::string& substring);

    /**
     * @brief 把int64_t的数组转为float类型的数组
     * @param input 输入是int64_t的数组
     * @return
     */
    std::vector<float> convert_int64_to_float(const std::vector<int64_t>& input);
}

#endif //BATMANINFER_UTILS_H
