//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_BI_TENSOR_SHAPE_HPP
#define BATMANINFER_BI_TENSOR_SHAPE_HPP

#include "data/core/dimensions.hpp"
#include <data/core/utils/misc/utils.hpp>

namespace BatmanInfer {
    class BITensorShape: public BIDimensions<size_t> {
    public:
        /**
         * @brief 初始化维度
         * @tparam Ts
         * @param dims
         */
        template <typename... Ts>
        BITensorShape(Ts... dims) : BIDimensions{dims...}
        {
            // 初始化不确定的维度为1
            if (_num_dimensions > 0)
                std::fill(_id.begin() + _num_dimensions, _id.end(), 1);

            // 移出尾部为1的点
            apply_dimension_correction();
        }

        /**
         * @brief 拷贝构造函数
         */
        BITensorShape(const BITensorShape &) = default;

        BITensorShape &operator=(const BITensorShape &) = default;

        BITensorShape(BITensorShape &&) = default;

        BITensorShape &operator=(BITensorShape &&) = default;

        ~BITensorShape() = default;

        /**
         * @brief 张量形状广播（Broadcasting）
         * 1， 广播的目的是让两个形状不同的张量能够进行数学运算（如加法、减法、乘法等）
         *
         * 2. 如果两个张量的形状是 广播兼容的，则返回广播后的形状
         * 广播兼容性规则：
         *   两个张量的每个维度要么相等，要么其中一个维度为 1。
         *   如果某个维度为 1，则该维度会被扩展（虚拟复制）以匹配另一个张量的维度。
         *   如果两个维度既不相等，也都不为 1，则它们 不兼容。
         *
         * 3. 如果两个形状是广播兼容的
         * * 广播后的形状中，每个维度的大小是原始两个维度中较大的那个值（max(dim1, dim2)）
         * 张量 A 的形状是 [4, 1, 3]
         * 张量 B 的形状是 [1, 5, 3]
         * 广播后的形状是 [4, 5, 3]，因为：
         * * 第 1 维：max(4, 1) = 4
         * * 第 2 维：max(1, 5) = 5
         * * 第 3 维：max(3, 3) = 3
         * @tparam Shapes
         * @param shapes
         * @return
         */
        template <typename... Shapes>
        static BITensorShape broadcast_shape(const Shapes &...shapes) {
            // 初始化广播形状
            BITensorShape bc_shape;

            auto broadcast = [&bc_shape](const BITensorShape &other)
            {
                // 当前广播形状为空，直接将 A 的形状赋值给 bc_shape
                if (bc_shape.num_dimensions() == 0)
                    bc_shape = other;
                else if (other.num_dimensions() != 0) {
                    // 如果 other 的形状不为空，则需要与当前的 bc_shape 进行广播计算
                    for (size_t d = 0; d < BITensorShape::num_max_dimensions; ++d) {
                        const size_t dim_min = std::min(bc_shape[d], other[d]);
                        const size_t dim_max = std::max(bc_shape[d], other[d]);

                        // 如果发现两个维度既不相等，也都不为 1，则广播失败，返回 [0]
                        if ((dim_min != 1) && (dim_min != dim_max)) {
                            // 将 bc_shape 设置为特殊值 [0]（表示广播失败），并退出循环
                            bc_shape = BITensorShape{0U};
                            break;
                        }

                        bc_shape.set(d, dim_max);
                    }
                }
            };

            misc::utility::for_each(broadcast, shapes...);

            return bc_shape;
        }

    private:
        /**
         * @brief 从维度数量中移除大小为1的尾随维度
         */
        void apply_dimension_correction() {
            for (int i = static_cast<int>(_num_dimensions) - 1; i > 0; --i) {
                if (_id[i] == 1)
                    --_num_dimensions;
                else
                    break;
            }
        }
    };
}

#endif //BATMANINFER_BI_TENSOR_SHAPE_HPP
