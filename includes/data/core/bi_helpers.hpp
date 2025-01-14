//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_HELPERS_HPP
#define BATMANINFER_BI_HELPERS_HPP

#include <data/core/bi_types.hpp>
#include <data/core/bi_window.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_error.h>

namespace BatmanInfer {
    /**
     * @brief 获取给定维度的索引。
     * @param data_layout_dimension 请求该索引的维度。
     * @return
     */
    inline size_t get_data_layout_dimension_index(const BIDataLayoutDimension &data_layout_dimension);

    /** Returns a static map used to find an index or dimension based on a data layout
  *
  * *** Layouts ***
  *
  * *** 4D ***
  * [N C H W]
  * [3 2 1 0]
  * [N H W C]
  *
  * * *** 5D ***
  * [N C D H W]
  * [4 3 2 1 0]
  * [N D H W C]
  */
    const std::vector<BIDataLayoutDimension> &get_layout_map();

    /**
     * @brief 迭代器: 由@ref execute_window_loop进行迭代更新
     */
    class BIIterator {
    public:
        /**
         * @brief 默认构造函数
         */
        constexpr BIIterator();

        /**
         * @brief 创建一个元数据的容器迭代器, 分配器包含Itensor
         * @param tensor 张量和迭代器关联
         * @param window 窗口将被用来迭代张量
         */
        explicit BIIterator(const BIITensor *tensor,
                            const BIWindow &window);

        /** 根据维度数量，步幅，缓冲区指针和窗口来创建容器迭代器
         * @brief
         * @param num_dims 维度的数量
         * @param strides  字节的步幅
         * @param buffer   数据缓冲区
         * @param offset   从缓冲区的开始到张量的第一个元素的字节偏移量。
         * @param window   将用于遍历张量的窗口。
         */
        explicit BIIterator(size_t num_dims,
                            const BIStrides &strides,
                            uint8_t *buffer,
                            size_t offset,
                            const BIWindow &window);

        /**
         * @brief 沿指定维度以与该维度相关联的步长值递增迭代器
         *
         * @warning 调用者有责任在到达某维度的末尾时调用 increment(dimension+1)，迭代器不会检查溢出
         *
         * @note 当递增维度 n 时，范围 (0, n-1) 内所有维度的坐标都会被重置。例如，如果你在遍历一个二维图像，每次改变行（维度 1）时，宽度（维度 0）的迭代器都会被重置为起始位置
         *
         * @param dimension
         */
        void increment(size_t dimension);

        /**
         * @brief 返回从第一个元素到迭代器当前位置信息的字节偏移量
         * @return  迭代器当前在字节中相对于第一个元素的位置。
         */
        constexpr size_t offset() const;

        /**
         * @brief 返回指向当前像素的指针。
         *
         * @warning 只有在迭代器是用 BIITensor 创建时才有效。
         *
         * @return equivalent to  buffer() + offset()
         */
        constexpr uint8_t *ptr() const;

        /**
         * @brief 将迭代器移动回指定维度的开头。
         * @param dimension
         */
        void reset(size_t dimension);

    private:
        /**
         * @brief 为具有指定维度数量、步幅、缓冲区指针和窗口的张量初始化一个容器迭代器。
         * @param num_dims 维度的数量
         * @param strides  字节的步幅
         * @param buffer   数据缓冲区
         * @param offset   从缓冲区的开始到张量的第一个元素的字节偏移量。
         * @param window   遍历张量的窗口
         */
        void initialize(size_t num_dims,
                        const BIStrides &strides,
                        uint8_t *buffer,
                        size_t offset,
                        const BIWindow &window);

        uint8_t *_ptr;

        class BIDimension {
        public:
            constexpr BIDimension() : _dim_start(0), _stride(0) {

            }

            size_t _dim_start;
            size_t _stride;
        };

        std::array<BIDimension, BICoordinates::num_max_dimensions> _dims;
    };

    /**
     * @brief 遍历传递的窗口，自动调整迭代器并为每个元素调用 lambda_function。
     *        它在每次迭代中将 x 和 y 位置传递给 lambda_function。
     * @tparam L
     * @tparam Ts
     * @param w 迭代的窗口
     * @param lambda_function 类型为 void(function)( const Coordinates & id ) 的函数在每次迭代时调用。
     *                        其中 id 表示要处理的项目的绝对坐标
     * @param iterators   在调用 lambda_function 之前，将由此函数更新的张量迭代器。
     */
    template<typename L, typename ... Ts>
    inline void execute_window_loop(const BIWindow &w,
                                    L &&lambda_function,
                                    Ts &&...iterators);

    /**
     * 将线性索引转换为n维坐标。
     * @param shape n维张量的形状。
     * @param index 线性索引指定第 i 个元素。
     * @return n-维度坐标
     */
    inline BICoordinates index2coords(const BITensorShape &shape, int index);

    /**
     * 坐标转index
     * @param shape 数据形状
     * @param coord
     * @return
     */
    inline int coords2index(const BITensorShape &shape, const BICoordinates &coord);

    /** Permutes given TensorShape according to a permutation vector
     *
     * @warning Validity of permutation is not checked
     *
     * @param[in, out] shape Shape to permute
     * @param[in]      perm  Permutation vector
     */
    inline void permute(BITensorShape &shape, const PermutationVector &perm)
    {
        BITensorShape shape_copy = shape;
        for (unsigned int i = 0; i < perm.num_dimensions(); ++i)
        {
            size_t dimension_val = (perm[i] < shape.num_dimensions()) ? shape_copy[perm[i]] : 1;
            shape.set(i, dimension_val, false, false); // Avoid changes in _num_dimension
        }
    }

    /** Wrap-around a number within the range 0 <= x < m
     *
     * @param[in] x Input value
     * @param[in] m Range
     *
     * @return the wrapped-around number
     */
    template <typename T>
    inline T wrap_around(T x, T m)
    {
        return x >= 0 ? x % m : (x % m + m) % m;
    }

}

#include <data/core/bi_helpers.inl>

#endif //BATMANINFER_BI_HELPERS_HPP
