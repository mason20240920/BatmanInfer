//
// Created by Mason on 2025/1/13.
//

#pragma once

#include <data/core/dimensions.hpp>
#include <data/core/bi_window.hpp>

#include <cpu/kernels/assembly/bi_nd_range.hpp>
#include <cassert>


/* 该文件包含 BatmanInfer 和 BatmanGemm 中使用的整型类型之间的映射关系
 * 这两个代码库为了保持模块化的需要，彼此独立定义了各自的类型，
 * 这些类型表示的是相似的信息。
 */

namespace BatmanGemm {
    //we want to unify the maximum number of dimensions used between BatmanGem and BatmanInfer library
    constexpr std::size_t ndrange_max = BatmanInfer::BIDimensions<unsigned int>::num_max_dimensions;

    using ndrange_t = BINDRange<ndrange_max>;
    using ndcoord_t = BINDCoordinate<ndrange_max>;

    /**
     * 将 `BatmanGemm::ndrange_t` 转换为 `BatmanInfer::BIWindow`
     *
     * 由于 `BINDRange<T>` 不包含起始位置的信息，我们在生成的 `BatmanInfer::BIWindow` 中
     * 将起始位置指定为零。
     * @param ndr
     * @return
     */
    inline BatmanInfer::BIWindow to_window(const ndrange_t &ndr) {
        BatmanInfer::BIWindow win;

        for (unsigned int i = 0; i != ndrange_max; ++i) {
            //populate the window with the dimensions of the NDRange
            win.set(i, BatmanInfer::BIWindow::BIDimension(0, ndr.get_size(i)));
        }

        return win;
    }

    inline BatmanInfer::BIWindow to_window(const ndcoord_t &ndc) {
        BatmanInfer::BIWindow win;

        for (unsigned int i = 0; i != ndrange_max; ++i) {
            const auto start = ndc.get_position(i);
            const auto size = ndc.get_size(i);
            const auto stop = start + size;

            //populate the window with the dimensions of the NDRange
            win.set(i, BatmanInfer::BIWindow::BIDimension(start, stop));
        }

        return win;
    }

    /**
     * 将arm_compute::Window转换为具有相同最大维度的arm_gemm::NDRange。
     * 应该注意的是，arm_compute::Window指定了start()和end()，
     * 而arm_gemm::ndrange_t只有大小，因此我们存储范围之间的差值。
     * @param win  我们要转换为arm_gemm::ndrange_t的arm_compute::Window
     * @return
     */
    inline ndrange_t to_ndrange(const BatmanInfer::BIWindow &win) {
        return {static_cast<unsigned int>(win[0].end() - win[0].start()),
                static_cast<unsigned int>(win[1].end() - win[1].start()),
                static_cast<unsigned int>(win[2].end() - win[2].start()),
                static_cast<unsigned int>(win[3].end() - win[3].start()),
                static_cast<unsigned int>(win[4].end() - win[4].start()),
                static_cast<unsigned int>(win[5].end() - win[5].start())};
    }

    inline ndcoord_t to_ndcoord(const BatmanInfer::BIWindow &win) {
        return {{static_cast<unsigned int>(win[0].start()), static_cast<unsigned int>(win[0].end() - win[0].start())},
                {static_cast<unsigned int>(win[1].start()), static_cast<unsigned int>(win[1].end() - win[1].start())},
                {static_cast<unsigned int>(win[2].start()), static_cast<unsigned int>(win[2].end() - win[2].start())},
                {static_cast<unsigned int>(win[3].start()), static_cast<unsigned int>(win[3].end() - win[3].start())},
                {static_cast<unsigned int>(win[4].start()), static_cast<unsigned int>(win[4].end() - win[4].start())},
                {static_cast<unsigned int>(win[5].start()), static_cast<unsigned int>(win[5].end() - win[5].start())}};
    }


}