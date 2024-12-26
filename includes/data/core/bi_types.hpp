//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_TYPES_HPP
#define BATMANINFER_BI_TYPES_HPP

/** The following symbols have been moved to:
 * half
 * PermutationVector
 * Format
 * DataType
 * DataLayout
 * DataLayoutDimension
 * PadStrideInfo
 * WeightFormat
 * Channel
 * DimensionRoundingType
 */
#include <data/core/core_types.hpp>

#include <data/core/bi_coordinates.hpp>

#include <data/bi_tensor_shape.hpp>

#include <string>

namespace BatmanInfer {
    /** Container for 2D border size */
    struct BIBorderSize
    {
        /** Empty border, i.e. no border */
        constexpr BIBorderSize() noexcept : top{0}, right{0}, bottom{0}, left{0}
        {
        }

        /** Border with equal size around the 2D plane */
        explicit constexpr BIBorderSize(unsigned int size) noexcept : top{size}, right{size}, bottom{size}, left{size}
        {
        }

        /** Border with same size for top/bottom and left/right */
        constexpr BIBorderSize(unsigned int top_bottom, unsigned int left_right)
                : top{top_bottom}, right{left_right}, bottom{top_bottom}, left{left_right}
        {
        }

        /** Border with different sizes */
        constexpr BIBorderSize(unsigned int top, unsigned int right, unsigned int bottom, unsigned int left)
                : top{top}, right{right}, bottom{bottom}, left{left}
        {
        }

        /** Check if the entire border is zero */
        constexpr bool empty() const
        {
            return top == 0 && right == 0 && bottom == 0 && left == 0;
        }

        /** Check if the border is the same size on all sides */
        constexpr bool uniform() const
        {
            return top == right && top == bottom && top == left;
        }

        /** Scale this border size.
         *
         * @param[in] scale Scale to multiply border size by.
         *
         * @return *this.
         */
        BIBorderSize &operator*=(float scale)
        {
            top *= scale;
            right *= scale;
            bottom *= scale;
            left *= scale;

            return *this;
        }

        /** Scale a copy of this border size.
         *
         * @param[in] scale Scale to multiply border size by.
         *
         * @return a scaled copy of this.
         */
        BIBorderSize operator*(float scale)
        {
            BIBorderSize size = *this;
            size *= scale;

            return size;
        }

        /** Check equality with another BIBorderSize struct
         *
         * @param[in] rhs other struct to check against
         *
         * @return true if they are equal
         */
        bool operator==(const BIBorderSize &rhs) const
        {
            return (top == rhs.top) && (right == rhs.right) && (bottom == rhs.bottom) && (left == rhs.left);
        }

        /** Check non-equality with another BIBorderSize struct
         *
         * @param[in] rhs other struct to check against
         *
         * @return true if they are different
         */
        bool operator!=(const BIBorderSize &rhs) const
        {
            return !(*this == rhs);
        }

        /** Limit this border size.
         *
         * @param[in] limit Border size to limit this border size to.
         */
        void limit(const BIBorderSize &limit)
        {
            top    = std::min(top, limit.top);
            right  = std::min(right, limit.right);
            bottom = std::min(bottom, limit.bottom);
            left   = std::min(left, limit.left);
        }

        unsigned int top;    /**< top of the border */
        unsigned int right;  /**< right of the border */
        unsigned int bottom; /**< bottom of the border */
        unsigned int left;   /**< left of the border */
    };

    /**
     * @brief 用于二维内边距大小的容器
     */
    using BIPaddingSize = BIBorderSize;

    /**
     * @brief 窗口有效区域的容器
     */
    struct BIValidRegion {

        BIValidRegion() : anchor{}, shape{}
        {
        }

        BIValidRegion(const BIValidRegion &) = default;
        BIValidRegion(BIValidRegion &&) = default;
        BIValidRegion &operator=(const BIValidRegion&) = default;
        BIValidRegion &operator=(BIValidRegion &&) = default;
        ~BIValidRegion() = default;

        BIValidRegion(const BICoordinates &an_anchor,
                    const BITensorShape &a_shape) : anchor{an_anchor}, shape{a_shape} {
            anchor.set_num_dimensions(std::max(anchor.num_dimensions(), shape.num_dimensions()));
        }

        BIValidRegion(const BICoordinates &an_anchor,
                    const BITensorShape &a_shape,
                    size_t num_dimensions) : anchor{an_anchor}, shape{a_shape} {
            ARM_COMPUTE_ERROR_ON(num_dimensions < std::max(anchor.num_dimensions(), shape.num_dimensions()));
            anchor.set_num_dimensions(num_dimensions);
        }

        /**
         * @brief 返回给定维度 @p d 的有效区域的起始位置。
         * @param d
         * @return
         */
        int start(unsigned int d) const {
            return anchor[d];
        }

        /**
         * @brief 返回给定维度 @p d 的有效区域的结束部分。
         * @param d
         * @return
         */
        int end(unsigned int d) const {
            return anchor[d] + shape[d];
        }

        /**
         * @brief 访问器用于设置一个维度的锚点和形状的值。
         * @param dimension 设置值的维度。
         * @param start 要在锚点中设置的维度值。
         * @param size
         * @return
         */
        BIValidRegion &set(size_t dimension,
                         int start,
                         size_t size) {
            anchor.set(dimension, start);
            shape.set(dimension, size);
            return *this;
        }

        /**
         * @brief Check whether two valid regions are equal
         * @param lhs
         * @param rhs
         * @return
         */
        inline friend bool operator==(const BIValidRegion &lhs, const BIValidRegion &rhs);

        /**
         * @brief 有效区域开始的锚点
         */
        BICoordinates anchor;

        /**
         * @brief 有效区域的形状
         */
        BITensorShape shape;
    };

    inline bool operator==(const BIValidRegion &lhs, const BIValidRegion &rhs) {
        return (lhs.anchor == rhs.anchor) && (lhs.shape == rhs.shape);
    }
}

#endif //BATMANINFER_BI_TYPES_HPP
