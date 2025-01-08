//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_UTILS_HPP
#define BATMANINFER_BI_UTILS_HPP

#include <data/core/bi_tensor_info.hpp>


namespace BatmanInfer {
    /**
     * @brief 根据提供的步幅和张量的维度，创建一个步幅对象。
     * @tparam T
     * @tparam Ts
     * @param info 张量信息对象，用于在未指定步幅时提供张量的形状。
     * @param stride_x 用于 X 维的步幅（以字节为单位）。
     * @param fixed_strides  用于从 Y 开始的更高维度的步幅（以字节为单位）。
     * @return 基于指定步幅的 Strides 对象。未指定的步幅根据张量形状和较低维度的步幅计算得出。
     */
    template<typename T, typename... Ts>
    inline BIStrides compute_strides(const BIITensorInfo &info, T stride_x, Ts &&...fixed_strides) {
        const BITensorShape &shape = info.tensor_shape();

        // Create strides object
        BIStrides strides(stride_x, fixed_strides...);

        for (size_t i = 1 + sizeof...(Ts); i < info.num_dimensions(); ++i) {
            strides.set(i, shape[i - 1] * strides[i - 1]);
        }

        return strides;
    }

    /** 根据提供的步幅和张量的维度，创建一个步幅对象。
     *
     * @param[in] info 张量信息对象，用于在未指定步幅时提供张量的形状。
    *
    * @return Strides object based on element size and tensor shape.
    */
    template<typename... Ts>
    inline BIStrides compute_strides(const BIITensorInfo &info) {
        return compute_strides(info, info.element_size());
    }

    /**
     * 检查张量是否存在空洞。
     *
     *  空洞被定义为张量中两个连续值之间的任何间隙。这可能是由于扩展了填充（padding）
     *  或操作了张量的步幅（stride）导致的。
     *
     * @param info 定义输入张量形状的张量信息对象。
     * @return
     */
    bool has_holes(const BIITensorInfo &info);

    /**
     *  检查张量是否存在空洞。
     * @param info 定义输入张量形状的张量信息对象。
     * @param dimension 要检查的最高维度。
     *
     * @note 此函数会检查所有维度中，直到并包括最高维度的空洞。
     *
     * @return
     */
    bool has_holes(const BIITensorInfo &info, size_t dimension);
}

#endif //BATMANINFER_BI_UTILS_HPP
