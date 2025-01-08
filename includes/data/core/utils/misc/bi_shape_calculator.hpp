//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_SHAPE_CALCULATOR_HPP
#define BATMANINFER_BI_SHAPE_CALCULATOR_HPP

#include <data/core/bi_tensor_info.hpp>

namespace BatmanInfer {
    namespace misc {
        namespace shape_calculator {

            /** 计算转置后的矩阵
             *
             * @param input input Input tensor info
             * @return
             */
            inline BITensorShape compute_transposed_shape(const BIITensorInfo &input) {
                BITensorShape shape_transposed{input.tensor_shape()};

                shape_transposed.set(0, input.dimension(1), false);
                shape_transposed.set(1, input.dimension(0), false);

                return shape_transposed;
            }

            /**
             * 计算一个输入张量的交叉形状
             * @param a
             * @param multi_interleave4x4_height
             * @param reinterpret_input_as_3d
             * @return
             */
            inline BITensorShape compute_interleaved_shape(const BIITensorInfo &a,
                                                           int multi_interleave4x4_height = 1,
                                                           bool reinterpret_input_as_3d = false) {
                // 输出的矩阵的形状: [ a_height * W, ceil(a_width / W) ] where W = 4 * multi_interleave4x4_height
                BI_COMPUTE_ERROR_ON(multi_interleave4x4_height < 1);
                // 计算交错宽度 W = 4 * multi_interleave4x4_height
                const int interleave_width = 4 * multi_interleave4x4_height;
                BITensorShape shape_interleaved_a{a.tensor_shape()};
                shape_interleaved_a.set(0, a.dimension(0) * interleave_width);
                if (reinterpret_input_as_3d) {
                    // 将输入张量的第1维和第2维合并为一个新的高度维度M
                    const int M = a.dimension(1) * a.dimension(2);
                    // 计算新的高度维度，按交错宽度进行分块，向上取整
                    const int height = std::ceil(M / static_cast<float>(interleave_width));
                    // 设置1维的高度
                    shape_interleaved_a.set(1, height);

                    // 当数据格式为 NHWC 且形状为 Nx1x1 时，
                    // 张量的维度数可能会自动设置为 1，而不是 3。
                    // 为了避免移除不存在的维度导致的错误，需要检查维度数是否大于 2。
                    if (shape_interleaved_a.num_dimensions() > 2)
                        shape_interleaved_a.remove_dimension(2);
                } else
                    shape_interleaved_a.set(1, std::ceil(a.dimension(1) / static_cast<float>(interleave_width)));

                return shape_interleaved_a;
            }
        }
    }
}

#endif //BATMANINFER_BI_SHAPE_CALCULATOR_HPP
