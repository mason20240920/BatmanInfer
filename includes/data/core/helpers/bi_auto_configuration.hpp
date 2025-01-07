//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_AUTO_CONFIGURATION_HPP
#define BATMANINFER_BI_AUTO_CONFIGURATION_HPP

#include <data/core/bi_i_tensor_info.hpp>

namespace BatmanInfer {
    /**
     * 使用另一个张量信息自动初始化张量信息。
     *
     * (COMPMID-6012) 此方法应与 ITensorInfo 中具有 setter 方法的字段保持同步。
     *
     * @param info_sink 用于检查和赋值的目标张量信息。
     * @param info_source 用于赋值的源张量信息。
     * @return 如果张量信息已被初始化，则返回 true。
     */
    inline bool auto_init_if_empty(BIITensorInfo &info_sink, const BIITensorInfo &info_source) {
        if (info_sink.tensor_shape().total_size() == 0) {
            info_sink.set_data_type(info_source.data_type());
            info_sink.set_num_channels(info_source.num_channels());
            info_sink.set_tensor_shape(info_source.tensor_shape());
            info_sink.set_quantization_info(info_source.quantization_info());
            info_sink.set_are_values_constant(info_source.are_values_constant());
            return true;
        }
        return false;
    }
}

#endif //BATMANINFER_BI_AUTO_CONFIGURATION_HPP
