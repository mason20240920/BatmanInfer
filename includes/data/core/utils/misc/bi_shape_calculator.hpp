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
        }
    }
}

#endif //BATMANINFER_BI_SHAPE_CALCULATOR_HPP
