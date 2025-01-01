//
// Created by Mason on 2025/1/1.
//

#ifndef BATMANINFER_BI_WINDOW_HPP
#define BATMANINFER_BI_WINDOW_HPP

#include <data/core/dimensions.hpp>
#include <data/core/bi_coordinates.hpp>
#include <data/core/utils/math/math.hpp>
#include <data/core/utils/misc/utils.hpp>
#include <data/core/bi_i_tensor_info.hpp>

namespace BatmanInfer {
    /**
     * @brief 描述多维执行窗口
     */
    class BIWindow {
    public:
        /**
         * @brief 维度0的别名: x
         */
        static constexpr size_t DimX = 0;

        static constexpr size_t DimY = 1;

        static constexpr size_t DimZ = 2;

        static constexpr size_t DimW = 3;
        /**
         * @brief 维度0的别名: v
         */
        static constexpr size_t DimV = 4;

        /**
         * @brief 默认构造函数
         *        创建一个包含单独元素的窗口
         */
        constexpr BIWindow() :
                _dims(),
                _is_broadcasted(misc::utility::generate_array<bool, BICoordinates::num_max_dimensions, false>::value) {

        }

        /**
         * @brief 拷贝函数
         * @param src
         */
        BIWindow(const BIWindow &src);

        BIWindow &operator=(const BIWindow &rhs);

        class BIDimension {
        public:
            /**
             * @brief Constructor, by default creates a dimension of 1.
             * @param start
             * @param end
             * @param step 在迭代时在维度的两个元素之间步数
             */
            constexpr BIDimension(int start = 0,
                                  int end = 1,
                                  int step = 1) : _start(start), _end(end), _step(step) {

            }

            BIDimension(const BIDimension &d) = default;

            BIDimension &operator=(const BIDimension &d) = default;

            constexpr int start() const {
                return _start;
            }

            constexpr int end() const {
                return _end;
            }

            constexpr int step() const {
                return _step;
            }

            void set_step(int step) {
                _step = step;
            }

            void set_end(int end) {
                _end = end;
            }


            /**
             * @brief 是否两个维度相同
             * @param lhs
             * @param rhs
             * @return
             */
            friend bool operator==(const BIDimension &lhs, const BIDimension &rhs) {
                return (lhs._start == rhs._start) && (lhs._end == rhs._end) && (lhs._step == rhs._step);
            }

        private:
            /**
             * @brief 维度的开始
             */
            int _start;
            /**
             * @brief 维度的结束
             */
            int _end;

            int _step;
        };

        /**
         * @brief 只读访问窗口的指定维度
         *
         * @note 前置条件：dimension < Coordinates::num_max_dimensions
         *
         * @param dimension 要访问的维度
         * @return 请求的维度
         */
        constexpr const BIDimension &operator[](size_t dimension) const;

        /**
         * @brief 窗口的第一个维度
         * @return
         */
        constexpr const BIDimension &x() const {
            return _dims.at(BIWindow::DimX);
        }

        constexpr const BIDimension &y() const {
            return _dims.at(BIWindow::DimY);
        }

        constexpr const BIDimension &z() const {
            return _dims.at(BIWindow::DimZ);
        }

        /**
         * @brief 设置一个给定的维度的值
         * @param dimension 需要设置的维度
         * @param dim  设置维度的值
         */
        void set(size_t dimension,
                 const BIDimension &dim);

        /**
         * @brief 将维度设置为广播维度。
         * @param dimension
         */
        void set_broadcasted(size_t dimension);

        /**
         * @brief 返回一个维度是否已被广播
         * @param dimension
         * @return
         */
        bool is_broadcasted(size_t dimension) const;

/** Use the tensor's dimensions to fill the window dimensions.
     *
     * @param[in] shape           @ref TensorShape to copy the dimensions from.
     * @param[in] first_dimension Only copy dimensions which are greater or equal to this value.
     */
        void use_tensor_dimensions(const BITensorShape &shape, size_t first_dimension = BIWindow::DimX);

        /** Shift the values of a given dimension by the given shift_value
         *
         * @param[in] dimension   The dimension to shift
         * @param[in] shift_value Value to shift the start and end values of.
         */
        void shift(size_t dimension, int shift_value);

        /** Shift down all the dimensions of a window starting from the specified dimension.
         *
         * new_dims[i] = old_dims[i]             for all i < start_dim.
         * new_dims[i] = old_dims[i+shift_value] for all i >= start_dim.
         *
         * @param[in] shift_value Number of dimensions to shift the window by.
         * @param[in] start_dim   The dimension from which the dimensions start to shift.
         *
         * @return The window with the shifted dimensions.
         */
        BIWindow shift_dimensions(unsigned int shift_value, unsigned int start_dim = 0) const;

        /** Adjust the start or end of a given dimension by the given value
         *
         * @param[in] dimension    The dimension to adjust
         * @param[in] adjust_value The adjusted value.
         * @param[in] is_at_start  The flag to indicate whether adjust the start or end of the dimension.
         */
        void adjust(size_t dimension, int adjust_value, bool is_at_start);

        /** Scale the values of a given dimension by the given scale_value
         *
         * @note The end of the window is rounded up to be a multiple of step after the scaling.
         *
         * @param[in] dimension   The dimension to scale
         * @param[in] scale_value Value to scale the start, end and step values of.
         */
        void scale(size_t dimension, float scale_value);

        /** Set the step of a given dimension.
         *
         * @param[in] dimension Dimension to update
         * @param[in] step      The new dimension's step value
         */
        void set_dimension_step(size_t dimension, int step);

        /** Will validate all the window's dimensions' values when asserts are enabled
         *
         * No-op when asserts are disabled
         */
        void validate() const;

        /** Return the number of iterations needed to iterate through a given dimension
         *
         * @param[in] dimension The requested dimension
         *
         * @return The number of iterations
         */
        constexpr size_t num_iterations(size_t dimension) const;
        /** Return the total number of iterations needed to iterate through the entire window
         *
         * @return Number of total iterations
         */
        size_t num_iterations_total() const;
        /** Return the shape of the window in number of steps */
        BITensorShape shape() const;
        /** Split a window into a set of sub windows along a given dimension
         *
         * For example to split a window into 3 sub-windows along the Y axis, you would have to do:<br/>
         * Window sub0 = window.split_window( 1, 0, 3);<br/>
         * Window sub1 = window.split_window( 1, 1, 3);<br/>
         * Window sub2 = window.split_window( 1, 2, 3);<br/>
         *
         * @param[in] dimension Dimension along which the split will be performed
         * @param[in] id        Id of the sub-window to return. Must be in the range (0, total-1)
         * @param[in] total     Total number of sub-windows the window will be split into.
         *
         * @return The subwindow "id" out of "total"
         */
        BIWindow split_window(size_t dimension, size_t id, size_t total) const;
        /** First 1D slice of the window
         *
         * @return The first slice of the window.
         */
        BIWindow first_slice_window_1D() const
        {
            return first_slice_window<1>();
        };
        /** First 2D slice of the window
         *
         * @return The first slice of the window.
         */
        BIWindow first_slice_window_2D() const
        {
            return first_slice_window<2>();
        };
        /** First 3D slice of the window
         *
         * @return The first slice of the window.
         */
        BIWindow first_slice_window_3D() const
        {
            return first_slice_window<3>();
        };
        /** First 4D slice of the window
         *
         * @return The first slice of the window.
         */
        BIWindow first_slice_window_4D() const
        {
            return first_slice_window<4>();
        };
        /** Slide the passed 1D window slice.
         *
         * If slice contains the last slice then it will remain unchanged and false will be returned.
         *
         * @param[in,out] slice Current slice, to be updated to the next slice.
         *
         * @return true if slice contains a new slice, false if slice already contained the last slice
         */
        bool slide_window_slice_1D(BIWindow &slice) const
        {
            return slide_window_slice<1>(slice);
        }
        /** Slide the passed 2D window slice.
         *
         * If slice contains the last slice then it will remain unchanged and false will be returned.
         *
         * @param[in,out] slice Current slice, to be updated to the next slice.
         *
         * @return true if slice contains a new slice, false if slice already contained the last slice
         */
        bool slide_window_slice_2D(BIWindow &slice) const
        {
            return slide_window_slice<2>(slice);
        }
        /** Slide the passed 3D window slice.
         *
         * If slice contains the last slice then it will remain unchanged and false will be returned.
         *
         * @param[in,out] slice Current slice, to be updated to the next slice.
         *
         * @return true if slice contains a new slice, false if slice already contained the last slice
         */
        bool slide_window_slice_3D(BIWindow &slice) const
        {
            return slide_window_slice<3>(slice);
        }
        /** Slide the passed 4D window slice.
         *
         * If slice contains the last slice then it will remain unchanged and false will be returned.
         *
         * @param[in,out] slice Current slice, to be updated to the next slice.
         *
         * @return true if slice contains a new slice, false if slice already contained the last slice
         */
        bool slide_window_slice_4D(BIWindow &slice) const
        {
            return slide_window_slice<4>(slice);
        }
        /** Collapse the dimensions between @p first and @p last if possible.
         *
         * A dimension is collapsable if it starts from 0 and matches the corresponding dimension in the full_window
         *
         * @param[in]  full_window   Full window @p window has been created from.
         * @param[in]  first         Start dimension into which the following are collapsed.
         * @param[in]  last          End (exclusive) dimension to collapse.
         * @param[out] has_collapsed (Optional) Whether the window was collapsed.
         *
         * @return Collapsed window.
         */
        BIWindow
        collapse_if_possible(const BIWindow &full_window, size_t first, size_t last, bool *has_collapsed = nullptr) const;

        /** Collapse the dimensions higher than @p first if possible.
         *
         * A dimension is collapsable if it starts from 0 and matches the corresponding dimension in the full_window
         *
         * @param[in]  full_window   Full window @p window has been created from.
         * @param[in]  first         Start dimension into which the following are collapsed.
         * @param[out] has_collapsed (Optional) Whether the window was collapsed.
         *
         * @return Collapsed window.
         */
        BIWindow collapse_if_possible(const BIWindow &full_window, size_t first, bool *has_collapsed = nullptr) const
        {
            return collapse_if_possible(full_window, first, BICoordinates::num_max_dimensions, has_collapsed);
        }

        /** Collapse the dimensions between @p first and @p last.
         *
         * A dimension is collapsable if it starts from 0 and matches the corresponding dimension in the full_window
         *
         * @param[in] full_window Full window @p window has been created from.
         * @param[in] first       Start dimension into which the following are collapsed.
         * @param[in] last        End (exclusive) dimension to collapse.
         *
         * @return Collapsed window if successful.
         */
        BIWindow collapse(const BIWindow &full_window, size_t first, size_t last = BICoordinates::num_max_dimensions) const;

        /** Don't advance in the dimension where @p shape is less equal to 1.
         *
         * @param[in] shape A TensorShape.
         *
         * @return Broadcast window.
         */
        BIWindow broadcast_if_dimension_le_one(const BITensorShape &shape) const;

        /** Don't advance in the dimension where shape of @p info is less equal to 1.
         *
         * @param[in] info An ITensorInfo.
         *
         * @return Broadcast window.
         */
        BIWindow broadcast_if_dimension_le_one(const BIITensorInfo &info) const
        {
            return broadcast_if_dimension_le_one(info.tensor_shape());
        }
        /** Friend function that swaps the contents of two windows
         *
         * @param[in] lhs First window to swap.
         * @param[in] rhs Second window to swap.
         */
        friend void swap(BIWindow &lhs, BIWindow &rhs);
        /** Check whether two Windows are equal.
         *
         * @param[in] lhs LHS window
         * @param[in] rhs RHS window
         *
         * @return True if the given windows are the same.
         */
        friend bool operator==(const BIWindow &lhs, const BIWindow &rhs);

    private:
        /** First slice of the window
         *
         * @return The first slice of the window.
         */
        template <unsigned int window_dimension>
        BIWindow first_slice_window() const;

        /** Slide the passed window slice.
         *
         * If slice contains the last slice then it will remain unchanged and false will be returned.
         *
         * @param[in,out] slice Current slice, to be updated to the next slice.
         *
         * @return true if slice contains a new slice, false if slice already contained the last slice
         */
        template <unsigned int window_dimension>
        bool slide_window_slice(BIWindow &slice) const;

    private:
        /**
         * @brief 每个维度的信息，总共有6个维度
         */
        std::array<BIDimension, BICoordinates::num_max_dimensions> _dims;
        /**
         * @brief 每个维度是否广播信息，总共有六个维度
         */
        std::array<bool, BICoordinates::num_max_dimensions>        _is_broadcasted;
    };
}
#include "bi_window.inl"
#endif //BATMANINFER_BI_WINDOW_HPP
