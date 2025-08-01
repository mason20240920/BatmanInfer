//
// Created by Mason on 2025/1/15.
//

#pragma once

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <runtime/cpp/functions/bi_cpp_split.hpp>
#include <runtime/bi_i_function.hpp>
#include <runtime/neon/functions/bi_ne_slice.hpp>

#include <memory>
#include <vector>

namespace BatmanInfer {
    /**
     * 沿给定轴分割张量的基本函数
     */
    class BINESplit : public BICPPSplit<BINESlice> {
    public:
        /** NESplit
        *
        * Valid data layouts:
        * - All
        *
        * Valid data type configurations:
        * |src            |dst            |
        * |:--------------|:--------------|
        * |All            |All            |
        *
        */
        void run();
    };
}