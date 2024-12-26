//
// Created by Mason on 2024/12/26.
//

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    const std::vector<BIDataLayoutDimension> &get_layout_map()
    {
        constexpr BIDataLayoutDimension W = BIDataLayoutDimension::WIDTH;
        constexpr BIDataLayoutDimension H = BIDataLayoutDimension::HEIGHT;
        constexpr BIDataLayoutDimension C = BIDataLayoutDimension::CHANNEL;
        constexpr BIDataLayoutDimension D = BIDataLayoutDimension::DEPTH;
        constexpr BIDataLayoutDimension N = BIDataLayoutDimension::BATCHES;

//        static const std::map<DataLayout, std::vector<DataLayoutDimension>> layout_map = {
//                {DataLayout::NDHWC, {C, W, H, D, N}},
//                {DataLayout::NCDHW, {W, H, D, C, N}},
//                {DataLayout::NHWC, {C, W, H, N}},
//                {DataLayout::NCHW, {W, H, C, N}}};

        return {C, W, H, N};
    }
}