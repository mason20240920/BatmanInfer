//
// Created by Mason on 2024/12/26.
//

#include <runtime/bi_allocator.hpp>
#include <data/core/bi_tensor_info.hpp>

using namespace BatmanInfer;

namespace {
    /**
     * @brief 验证子张量形状是否与父张量形状兼容。
     * @param parent_info
     * @param child_info
     * @param coords
     * @return
     */
    bool validate_sub_tensor_shape(const BITensorInfo &parent_info,
                                   const BITensorInfo &child_info,
                                   const BICoordinates &coords) {
        // 初始化验证结果为 true，假设子张量形状是有效的
        bool is_valid = true;

        // 获取父张量和子张量的形状
        const BITensorShape &parent_shape = parent_info.tensor_shape();
        const BITensorShape &child_shape = child_info.tensor_shape();

        // 获取父张量和子张量的维度数
        const size_t parent_dims = parent_info.num_dimensions();
        const size_t child_dims = child_info.num_dimensions();

        // 如果子张量的维度数小于或等于父张量的维度数
        if (child_dims <= parent_dims) {
            // 从子张量的最高维度（最后一个维度）开始逐个验证
            for (size_t num_dimensions = child_dims; num_dimensions > 0; --num_dimensions) {
                // 计算子张量在当前维度的结束位置
                const size_t child_dim_size = coords[num_dimensions - 1] + child_shape[num_dimensions - 1];

                // 检查以下条件：
                // 1. 子张量的起始坐标不能为负值。
                // 2. 子张量的结束位置不能超过父张量对应维度的大小。
                if ((coords[num_dimensions - 1] < 0) || (child_dim_size > parent_shape[num_dimensions - 1])) {
                    is_valid = false;
                    break;
                }
            }
        } else
            // 如果子张量的维度数大于父张量的维度数，直接认为无效
            is_valid = false;

        return is_valid;
    }
}