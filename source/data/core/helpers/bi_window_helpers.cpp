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

    BIWindow
    calculate_max_window(const BITensorShape &shape, const BISteps &steps, bool skip_border, BIBorderSize border_size) {
        if (!skip_border) {
            border_size = BIBorderSize(0);
        }

        BIWindow window;

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
}