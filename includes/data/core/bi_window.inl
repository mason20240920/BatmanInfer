
#ifndef BATMANINFER_BI_WINDOW_INL
#define BATMANINFER_BI_WINDOW_INL

namespace BatmanInfer {
    inline BIWindow::BIWindow(const BIWindow &src)
            : _dims(),
              _is_broadcasted(misc::utility::generate_array<bool, BICoordinates::num_max_dimensions, false>::value) {
        for (size_t i = 0; i < BICoordinates::num_max_dimensions; ++i) {
            set(i, src[i]);
            _is_broadcasted[i] = src.is_broadcasted(i);
        }
    }

    inline BIWindow &BIWindow::operator=(const BatmanInfer::BIWindow &rhs) {
        BIWindow tmp(rhs);
        swap(*this, tmp);
        return *this;
    }

    inline constexpr const BIWindow::BIDimension &BIWindow::operator[](size_t dimension) const {
        // Precondition: dimension < Coordinates::num_max_dimensions
        return _dims.at(dimension);
    }

    inline bool BIWindow::is_broadcasted(size_t dimension) const {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        return _is_broadcasted[dimension];
    }

    inline void BIWindow::set(size_t dimension, const BatmanInfer::BIWindow::BIDimension &dim) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        _dims[dimension] = dim;
    }

    inline void BIWindow::set_broadcasted(size_t dimension) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        set(dimension, BIDimension(0, 0, 0));
        _is_broadcasted[dimension] = true;
    }

    inline BIWindow BIWindow::collapse_if_possible(const BIWindow &full_window,
                                                   const size_t first,
                                                   const size_t last,
                                                   bool *has_collapsed) const {
        // 创建一个新的窗口对象 collapsed，初始化为当前窗口的副本
        BIWindow collapsed(*this);

        // 初始化变量 is_collapsable，用于判断维度是否可以折叠
        bool is_collapsable = true;

        // 初始化 collapsed_end 为第一个维度的结束位置（end 值）
        int collapsed_end = _dims[first].end();

        // 遍历从 first+1 到 last 的所有维度，检查是否满足折叠条件。
        for (size_t d = first + 1; d < last; ++d) {
            /**
             * 检查当前维度是否满足以下折叠条件：
             * 1. 当前维度的起始位置为 0
             * 2. full_window 对应维度的起始位置也为 0
             * 3. 当前维度的步长 (step) 小于等于 1
             * 4. full_window 对应维度的结束位置与当前维度的结束位置相同
             */
            is_collapsable = (_dims[d].start() == 0) && (full_window[d].start() == 0) && (_dims[d].step() <= 1) &&
                             (full_window[d].end() == _dims[d].end());

            // 如果维度满足折叠条件，则将当前维度的结束位置乘入 collapsed_end
            collapsed_end *= _dims[d].end();
        }

        // 如果所有维度都满足折叠条件，则进行折叠操作。
        if (is_collapsable) {
            // 更新第一个维度的结束位置为 collapsed_end
            collapsed._dims.at(first).set_end(collapsed_end);
            // 将从 first+1 到 last 的所有维度设置为空维度（BIDimension()）。
            for (size_t d = first + 1; d < last; ++d)
                collapsed.set(d, BIDimension());
        }

        // 如果 has_collapsed 指针不为空，则将折叠结果（是否成功折叠）写入该指针
        if (has_collapsed != nullptr)
            *has_collapsed = is_collapsable;


        return collapsed;
    }

    inline BIWindow BIWindow::shift_dimensions(unsigned int shift_value, unsigned int start_dim) const {
        BIWindow shifted_window;
        size_t n = 0;

        for (; n < start_dim; ++n) {
            shifted_window.set(n, _dims[n]);
        }

        for (; n < (BICoordinates::num_max_dimensions - shift_value); n++) {
            shifted_window.set(n, _dims[n + shift_value]);
        }

        return shifted_window;
    }

    inline BIWindow BIWindow::collapse(const BIWindow &full_window, const size_t first, const size_t last) const {
        bool has_collapsed = false;
        BIWindow collapsed = collapse_if_possible(full_window, first, last, &has_collapsed);
        // Make sure that the window has collapsed
        BI_COMPUTE_ERROR_ON(!has_collapsed);
        return collapsed;
    }

    inline BIWindow BIWindow::broadcast_if_dimension_le_one(const BITensorShape &shape) const {
        BIWindow broadcastWin(*this);
        for (size_t d = 0; d < BITensorShape::num_max_dimensions; ++d) {
            if (shape[d] <= 1) {
                broadcastWin.set_broadcasted(d);
            }
        }
        return broadcastWin;
    }

    inline void BIWindow::shift(size_t dimension, int shift_value) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        BIWindow::BIDimension &d = _dims[dimension];
        d = BIWindow::BIDimension(d.start() + shift_value, d.end() + shift_value, d.step());
    }

    inline void BIWindow::adjust(size_t dimension, int adjust_value, bool is_at_start) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        BIWindow::BIDimension &d = _dims[dimension];

        if (is_at_start) {
            d = BIWindow::BIDimension(d.start() + adjust_value, d.end(), d.step());
        } else {
            d = BIWindow::BIDimension(d.start(), d.end() + adjust_value, d.step());
        }
    }

    inline void BIWindow::scale(size_t dimension, float scale_value) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        BIWindow::BIDimension &d = _dims[dimension];
        const int scaled_step = d.step() * scale_value;
        const int scaled_start = d.start() * scale_value;
        const int scaled_diff = (d.end() - d.start()) * scale_value;
        const int scaled_end = scaled_start + ceil_to_multiples(scaled_diff, scaled_step);

        d = BIWindow::BIDimension(scaled_start, scaled_end, scaled_step);
    }

    inline void BIWindow::set_dimension_step(size_t dimension, int step) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);
        _dims[dimension].set_step(step);
    }

    inline void BIWindow::validate() const {
        for (size_t i = 0; i < BICoordinates::num_max_dimensions; ++i) {
            BI_COMPUTE_ERROR_ON(_dims[i].end() < _dims[i].start());
            BI_COMPUTE_ERROR_ON(
                    (_dims[i].step() != 0) && (((_dims[i].end() - _dims[i].start()) % _dims[i].step()) != 0));
        }
    }

    inline constexpr size_t BIWindow::num_iterations(size_t dimension) const {
        // Precondition: dimension < Coordinates::num_max_dimensions
        // Precondition: (end - start) % step == 0
        return (_dims.at(dimension).end() - _dims.at(dimension).start()) / _dims.at(dimension).step();
    }

    inline BIWindow BIWindow::split_window(size_t dimension, size_t id, size_t total) const {
        BI_COMPUTE_ERROR_ON(id >= total);
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);

        BIWindow out;

        for (size_t d = 0; d < BICoordinates::num_max_dimensions; ++d) {
            if (d == dimension) {
                int start = _dims[d].start();
                int end = _dims[d].end();
                const int step = _dims[d].step();

                const int num_it = num_iterations(d);
                const int rem = num_it % total;
                int work = num_it / total;

                int it_start = work * id;

                if (int(id) < rem) {
                    ++work;
                    it_start += id;
                } else {
                    it_start += rem;
                }

                start += it_start * step;
                end = std::min(end, start + work * step);

                out.set(d, BIDimension(start, end, step));
            } else {
                out.set(d, _dims[d]);
            }
        }

        return out;
    }

    template<unsigned int window_dimension>
    inline bool BIWindow::slide_window_slice(BIWindow &slice) const {
        for (unsigned int n = window_dimension; n < BICoordinates::num_max_dimensions; ++n) {
            // Did we reach the end of this dimension?
            const int v = slice._dims[n].start() + 1;

            if (v < _dims[n].end()) {
                // No: increment
                slice._dims[n] = BIDimension(v, v + 1, 1);

                // Reset lower dimensions:
                for (unsigned int lower = window_dimension; lower < n; ++lower) {
                    slice._dims[lower] = BIDimension(_dims[lower].start(), _dims[lower].start() + 1, 1);
                }
                return true;
            }
        }

        // It was the last slice
        return false; // Iteration over
    }

    template<unsigned int window_dimension>
    inline BIWindow BIWindow::first_slice_window() const {
        BIWindow slice;

        std::copy_n(_dims.begin(), window_dimension, slice._dims.begin());

        //Initialise higher dimensions to be the first slice.
        for (unsigned int n = window_dimension; n < BICoordinates::num_max_dimensions; ++n) {
            slice._dims[n] = BIDimension(_dims[n].start(), _dims[n].start() + 1, 1);
        }

        return slice;
    }

    inline void BIWindow::use_tensor_dimensions(const BITensorShape &shape, size_t first_dimension) {
        for (unsigned int n = first_dimension; n < shape.num_dimensions(); ++n) {
            set(n, BIWindow::BIDimension(0, std::max(shape[n], static_cast<size_t>(1))));
        }
    }

    inline BITensorShape BIWindow::shape() const {
        BITensorShape shape;
        for (size_t d = 0; d < BITensorShape::num_max_dimensions; ++d) {
            shape.set(d, (_dims[d].end() - _dims[d].start()) / _dims[d].step());
        }
        return shape;
    }

    inline size_t BIWindow::num_iterations_total() const {
        size_t total = 1;
        for (size_t d = 0; d < BICoordinates::num_max_dimensions; ++d) {
            total *= num_iterations(d);
        }
        return total;
    }

    inline void swap(BIWindow &lhs, BIWindow &rhs) {
        lhs._dims.swap(rhs._dims);
    }

    inline bool operator==(const BIWindow &lhs, const BIWindow &rhs) {
        return (lhs._dims == rhs._dims) && (lhs._is_broadcasted == rhs._is_broadcasted);
    }

}

#endif