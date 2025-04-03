//
// Created by Mason on 2025/1/7.
//

#include <data/core/helpers/bi_window_helpers.hpp>

namespace BatmanInfer {
    BIWindow calculate_max_window(const BIValidRegion &valid_region,
                                  const BISteps &steps,
                                  bool skip_border,
                                  BIBorderSize border_size) {
        if (!skip_border)
            border_size = BIBorderSize(0);

        const BICoordinates &anchor = valid_region.anchor;
        const BITensorShape &shape = valid_region.shape;

        BIWindow window;

        window.set(0, BIWindow::BIDimension(anchor[0] + border_size.left,
                                            anchor[0] + border_size.left + ceil_to_multiples(std::max(0,
                                                    static_cast<int>(shape[0]) -
                                                    static_cast<int>(border_size.left) -
                                                    static_cast<int>(border_size.right)),
                                                steps[0]),
                                            steps[0]));

        size_t n = 1;

        if (anchor.num_dimensions() > 1) {
            window.set(1,
                       BIWindow::BIDimension(
                           // Skip the border above the image
                           anchor[1] + border_size.top,
                           // Skip the border below the image
                           anchor[1] + border_size.top +
                           ceil_to_multiples(
                               std::max(0, static_cast<int>(shape[1]) - static_cast<int>(border_size.top) -
                                           static_cast<int>(border_size.bottom)),
                               steps[1]),
                           steps[1]));

            ++n;
        }

        if (anchor.num_dimensions() > 2) {
            window.set(2, BIWindow::BIDimension(anchor[2], std::max<size_t>(1, shape[2]), steps[2]));

            ++n;
        }

        for (; n < anchor.num_dimensions(); ++n) {
            window.set(n, BIWindow::BIDimension(anchor[n], std::max<size_t>(1, shape[n])));
        }

        for (; n < BICoordinates::num_max_dimensions; ++n) {
            window.set(n, BIWindow::BIDimension(0, 1));
        }

        return window;
    }

    /**
     * @brief 专门适配NEGather动态的函数
     * @param shape
     * @param steps
     * @param update_window
     * @return
     */
    void dynamic_calculate_max_window(const BITensorShape &shape, const BISteps &steps, BIWindow &update_window) {
        size_t n = 1;
        if (shape.num_dimensions() > 1) {
            update_window.set(1, BIWindow::BIDimension(0, ceil_to_multiples(
                                                           std::max(0, static_cast<int>(shape[1])),
                                                           steps[1]),
                                                       steps[1]));
            ++n;
        }
        if (shape.num_dimensions() > 2) {
            update_window.set(2, BIWindow::BIDimension(0, std::max<size_t>(1, shape[2]), steps[2]));
            ++n;
        }

        for (; n < shape.num_dimensions(); ++n) {
            update_window.set(n, BIWindow::BIDimension(0, std::max<size_t>(1, shape[n])));
        }

        for (; n < BICoordinates::num_max_dimensions; ++n) {
            update_window.set(n, BIWindow::BIDimension(0, 1));
        }
    }

    void dynamic_origin_max_window(const BITensorShape &shape, const BISteps &steps, BIWindow &update_window) {
        // 设置第 0 维窗口范围
        update_window.set(0, BIWindow::BIDimension(0,
                                                   ceil_to_multiples(
                                                       std::max(
                                                           0, static_cast<int>(shape[0])),
                                                       steps[0]),
                                                   steps[0]));
        size_t n = 1;
        if (shape.num_dimensions() > 1) {
            update_window.set(1, BIWindow::BIDimension(0, ceil_to_multiples(
                                                           std::max(0, static_cast<int>(shape[1])),
                                                           steps[1]),
                                                       steps[1]));
            ++n;
        }
        if (shape.num_dimensions() > 2) {
            update_window.set(2, BIWindow::BIDimension(0, std::max<size_t>(1, shape[2]), steps[2]));
            ++n;
        }

        for (; n < shape.num_dimensions(); ++n) {
            update_window.set(n, BIWindow::BIDimension(0, std::max<size_t>(1, shape[n])));
        }

        for (; n < BICoordinates::num_max_dimensions; ++n) {
            update_window.set(n, BIWindow::BIDimension(0, 1));
        }
    }

    /**
     * 计算一个张量的最大滑动窗口（BIWindow），用于遍历张量时定义窗口的范围和步长
     * @param shape 表示张量的形状（BITensorShape），包含张量每个维度的大小
     * @param steps 每个维度的步长（BISteps），决定窗口在该维度上移动的间隔
     * @param skip_border 是否跳过边界（布尔值）。如果为 false，边界处理被忽略，border_size 被设置为 0。
     * @param border_size 边界大小（BIBorderSize），定义张量的上下左右需要跳过的边界宽度
     * @return
     */
    BIWindow
    calculate_max_window(const BITensorShape &shape, const BISteps &steps, bool skip_border, BIBorderSize border_size) {
        if (!skip_border) {
            border_size = BIBorderSize(0);
        }

        BIWindow window;

        // 设置第 0 维窗口范围
        window.set(0, BIWindow::BIDimension(
                       // Skip the border left of the image
                       border_size.left,
                       // Skip the border right of the image
                       // Make sure the window width is a multiple of the step size
                       border_size.left +
                       ceil_to_multiples(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) -
                                                     static_cast<int>(border_size.right)),
                                         steps[0]),
                       steps[0]));

        size_t n = 1;

        if (shape.num_dimensions() > 1) {
            window.set(1, BIWindow::BIDimension(
                           // Skip the border above the image
                           border_size.top,
                           // Skip the border below the image
                           border_size.top + ceil_to_multiples(std::max(0, static_cast<int>(shape[1]) -
                                                                           static_cast<int>(border_size.top) -
                                                                           static_cast<int>(border_size.bottom)),
                                                               steps[1]),
                           steps[1]));

            ++n;
        }

        if (shape.num_dimensions() > 2) {
            window.set(2, BIWindow::BIDimension(0, std::max<size_t>(1, shape[2]), steps[2]));

            ++n;
        }

        for (; n < shape.num_dimensions(); ++n) {
            window.set(n, BIWindow::BIDimension(0, std::max<size_t>(1, shape[n])));
        }

        for (; n < BICoordinates::num_max_dimensions; ++n) {
            window.set(n, BIWindow::BIDimension(0, 1));
        }

        return window;
    }

    std::pair<BIWindow, size_t> calculate_squashed_or_max_window(const BIITensorInfo &src) {
        const auto &shape = src.tensor_shape();
        const auto &strides = src.strides_in_bytes();
        const auto num_dimensions = src.num_dimensions();

        BIWindow win;
        size_t split_dimension = BIWindow::DimY;
        size_t dim = 0;
        size_t squashed_bytes = src.element_size();

        // Try to squash the low dimensions together.
        for (; dim < num_dimensions; ++dim) {
            if (strides[dim] != squashed_bytes) {
                break;
            }
            squashed_bytes *= shape[dim];
        }
        if (dim == num_dimensions) {
            const auto squashed_elements = squashed_bytes / src.element_size();
            split_dimension = BIWindow::DimX;
            // The input tensor can be interpreted as 1D array.
            win.set(0, BIWindow::BIDimension(0, squashed_elements, 1));
            for (dim = 1; dim < BICoordinates::num_max_dimensions; ++dim) {
                win.set(dim, BIWindow::BIDimension(0, 1, 1));
            }
        } else {
            // Generate the max window.
            for (dim = 0; dim < BICoordinates::num_max_dimensions; ++dim) {
                win.set(dim, BIWindow::BIDimension(0, shape[dim], 1));
            }
        }
        return std::make_pair(win, split_dimension);
    }

    std::pair<BIWindow, size_t> calculate_squashed_or_max_window(const BIITensorInfo &src0, const BIITensorInfo &src1) {
        const auto &shape0 = src0.tensor_shape();
        const auto &shape1 = src1.tensor_shape();
        const auto &strides0 = src0.strides_in_bytes();
        const auto &strides1 = src1.strides_in_bytes();
        const auto num_dimensions = std::max(src0.num_dimensions(), src1.num_dimensions());

        BIWindow win;
        size_t split_dimension = BIWindow::DimY;
        size_t dim = 0;

        size_t squashed_bytes = src0.element_size();

        // Try to squash the low dimensions together.
        for (; dim < num_dimensions; ++dim) {
            if (shape0[dim] != shape1[dim] || strides0[dim] != squashed_bytes || strides1[dim] != squashed_bytes) {
                break;
            }

            squashed_bytes *= shape0[dim];
        }

        if (dim == num_dimensions) {
            auto squashed_elements = squashed_bytes / src0.element_size();

            split_dimension = BIWindow::DimX;

            // The input tensors can be interpreted as 1D array.
            win.set(0, BIWindow::BIDimension(0, squashed_elements, 1));

            for (dim = 1; dim < BICoordinates::num_max_dimensions; ++dim) {
                win.set(dim, BIWindow::BIDimension(0, 1, 1));
            }
        } else {
            // Generates the max window.
            for (dim = 0; dim < BICoordinates::num_max_dimensions; ++dim) {
                win.set(dim, BIWindow::BIDimension(0, std::max(shape0[dim], shape1[dim]), 1));
            }
        }

        return std::make_pair(win, split_dimension);
    }

    BIWindow calculate_max_window_horizontal(const BIValidRegion &valid_region,
                                             const BISteps &steps,
                                             bool skip_border,
                                             BIBorderSize border_size) {
        if (skip_border) {
            border_size.top = 0;
            border_size.bottom = 0;
        } else {
            border_size.left = 0;
            border_size.right = 0;
        }

        const BICoordinates &anchor = valid_region.anchor;
        const BITensorShape &shape = valid_region.shape;

        BIWindow window;

        window.set(0, BIWindow::BIDimension(
                       // Skip the border left of the image
                       anchor[0] + border_size.left,
                       // Skip the border right of the image
                       // Make sure the window width is a multiple of the step size
                       anchor[0] + border_size.left +
                       ceil_to_multiples(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) -
                                                     static_cast<int>(border_size.right)),
                                         steps[0]),
                       steps[0]));

        size_t n = 1;

        if (anchor.num_dimensions() > 1) {
            window.set(1, BIWindow::BIDimension(
                           // Skip the border above the image
                           anchor[1] - border_size.top,
                           // Skip the border below the image
                           anchor[1] + shape[1] + border_size.bottom, 1));

            ++n;
        }

        for (; n < anchor.num_dimensions(); ++n) {
            window.set(n, BIWindow::BIDimension(anchor[n], std::max<size_t>(1, shape[n])));
        }

        for (; n < BICoordinates::num_max_dimensions; ++n) {
            window.set(n, BIWindow::BIDimension(0, 1));
        }

        return window;
    }
}
