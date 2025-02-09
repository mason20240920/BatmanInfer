//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    /**
     * 给定的归一化类型进行维度归一化计算
     * @param layout
     * @param info
     * @return 归一化维度信息
     */
    inline unsigned int get_normalization_dimension_index(BIDataLayout layout,
                                                          const BINormalizationLayerInfo &info) {
        const unsigned int width_idx = get_data_layout_dimension_index(layout, BIDataLayoutDimension::WIDTH);
        const unsigned int channel_idx = get_data_layout_dimension_index(layout, BIDataLayoutDimension::CHANNEL);
        return info.is_in_map() ? width_idx : channel_idx;
    }
}