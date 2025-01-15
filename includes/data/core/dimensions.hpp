//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_DIMENSIONS_H
#define BATMANINFER_DIMENSIONS_H

#include <data/core/bi_error.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>

namespace BatmanInfer {

    /**
     * @brief 用于指示窗口、张量形状和坐标的最大维度的常量值。
     */
    constexpr size_t MAX_DIMS = 6;

    template<typename T>
    class BIDimensions {
    public:
        /**
         * @brief 张量最大维度数
         */
        static constexpr size_t num_max_dimensions = MAX_DIMS;

        /**
         * @brief 初始化TensorShape
         * @tparam Ts
         * @param dims 初始化维度的值
         */
        template<typename... Ts>
        explicit BIDimensions(Ts... dims) : _id{{static_cast<T>(dims)...}}, _num_dimensions{sizeof...(dims)} {

        }

        /**
         * @brief 允许实例能被拷贝
         */
        BIDimensions(const BIDimensions &) = default;

        /**
         * @brief 允许实例被拷贝
         * @return
         */
        BIDimensions &operator=(const BIDimensions &) = default;

        /**
         * @brief 默认实例能被移动
         */
        BIDimensions(BIDimensions &&) = default;

        /**
         * @brief 允许实例被移动
         * @return
         */
        BIDimensions &operator=(BIDimensions &&) = default;

        /**
         * @brief 设置某一个维度的值
         * @param dimensions
         * @param value
         * @param increase_dim_unit
         */
        void set(size_t dimension,
                 T value,
                 bool increase_dim_unit = true) {
            BI_COMPUTE_ERROR_ON(dimension >= num_max_dimensions);
            _id[dimension] = value;
            // 如果新维度是1，就不要增加维度的数量。
            if (increase_dim_unit || value != 1)
                _num_dimensions = std::max(_num_dimensions, dimension + 1);
        }

        T x() const {
            return _id[0];
        }

        T y() const {
            return _id[1];
        }

        T z() const {
            return _id[2];
        }

        /**
         * @brief 通过步长增加给定的维度，避免溢出。
         *
         * @note 前提条件：dim < _num_dimensions
         *
         * @param dim 需要增加的维度
         * @param step 步骤以增加 @p dim。
         */
        void increment(size_t dim,
                       T step = 1) {
            BI_COMPUTE_ERROR_ON(dim >= _num_dimensions);
            // std::numeric_limits<T>::max(): 获取类型 T 能表示的最大值
            // _id[dim]: 当前维度的值
            if ((std::numeric_limits<T>::max() - _id[dim]) >= step)
                _id[dim] += step;
        }

        const T &operator[](size_t dimension) const {
            BI_COMPUTE_ERROR_ON(dimension >= num_max_dimensions);
            return _id[dimension];
        }

        T &operator[](size_t dimension) {
            BI_COMPUTE_ERROR_ON(dimension >= num_max_dimensions);
            return _id[dimension];
        }

        unsigned int num_dimensions() const {
            return _num_dimensions;
        }

        void set_num_dimensions(size_t num_dimensions) {
            _num_dimensions = num_dimensions;
        }

        /**
         * @brief 合并维度。
         * @param n 要压缩到的维度数量 @p first
         * @param first @n 以下的维度要合并
         */
        void collapse(const size_t n,
                      const size_t first = 0) {
            BI_COMPUTE_ERROR_ON(first + n > _id.size());

            const size_t last = std::min(_num_dimensions, first + n);

            if (last > (first + 1)) {
                // 合并维度到第一个里面
                _id[first] = std::accumulate(&_id[first],
                                             &_id[last],
                                             1,
                                             std::multiplies<T>());
                // 将剩余的维度向下移动。(后面元素向前移动)
                std::copy(&_id[last], &_id[num_max_dimensions], &_id[first + 1]);
                // 减少维度
                const size_t old_num_dimensions = _num_dimensions;
                _num_dimensions -= last - first - 1;
                // 用0填充其他维度
                std::fill(&_id[_num_dimensions], &_id[old_num_dimensions], 0);
            }
        }

        /**
         * @brief  从给的一个点进行合并
         * @param start
         */
        void collapse_from(size_t start) {
            BI__COMPUTE_ERROR_ON(start > num_dimensions());
            collapse(num_dimensions() - start, start);
        }

        void remove(size_t idx) {
            BI_COMPUTE_ERROR_ON(_num_dimensions < 1);
            if (idx >= _num_dimensions)
                return;

            std::copy(_id.begin() + idx + 1, _id.end(), _id.begin() + idx);
            _num_dimensions--;

            // 用0填充其他维度
            std::fill(_id.begin() + _num_dimensions, _id.end(), 0);
        }

        typename std::array<T, num_max_dimensions>::iterator begin() {
            return _id.begin();
        }

        typename std::array<T, num_max_dimensions>::const_iterator begin() const {
            return _id.begin();
        }

        typename std::array<T, num_max_dimensions>::const_iterator cbegin() const {
            return begin();
        }

        typename std::array<T, num_max_dimensions>::iterator end() {
            return _id.end();
        }

        typename std::array<T, num_max_dimensions>::const_iterator end() const {
            return _id.end();
        }

        typename std::array<T, num_max_dimensions>::const_iterator cend() const {
            return end();
        }

        virtual /** Collapses all dimensions to a single linear total size.
        *
        * @return The total tensor size in terms of elements.
       */
        size_t total_size() const {
            return std::accumulate(_id.begin(), _id.end(), 1, std::multiplies<size_t>());
        }


    protected:
        /**
         * @brief 保护构造函数
         */
        ~BIDimensions() = default;

        /**
         * @brief
         * @param T: 数组类型
         * @param num_max_dimensions: 编译时确定的常量, 表示数组大小(默认为6)
         */
        std::array<T, num_max_dimensions> _id;

        /**
         * @brief 默认维度的大小(大小为0)
         */
        size_t _num_dimensions{0};
    };

    template<typename T>
    inline bool operator==(const BIDimensions<T> &lhs, const BIDimensions<T> &rhs) {
        return ((lhs.num_dimensions() == rhs.num_dimensions()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
    }

    template<typename T>
    inline bool operator!=(const BIDimensions<T> &lhs, const BIDimensions<T> &rhs) {
        return !(lhs == rhs);
    }

}

#endif //BATMANINFER_DIMENSIONS_H
