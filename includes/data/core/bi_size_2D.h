//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_BI_SIZE_2D_H
#define BATMANINFER_BI_SIZE_2D_H

#include <cstddef>
#include <string>
#include <utility>

namespace BatmanInfer {

    /** Class for specifying the size of an image or rectangle */
    class Size2D
    {
    public:
        /** Default constructor */
        Size2D() = default;
        /** Constructor. Initializes "width" and "height" respectively with "w" and "h"
         *
         * @param[in] w Width of the image or rectangle
         * @param[in] h Height of the image or rectangle
         */
        Size2D(size_t w, size_t h) noexcept : width(w), height(h)
        {
        }
        /** The area of the image or rectangle calculated as (width * height)
         *
         * @return Area (width * height)
         *
         */
        size_t area() const
        {
            return (width * height);
        }

        bool operator==(const Size2D &other) const
        {
            return (width == other.width) && (height == other.height);
        }

        bool operator!=(const Size2D &other) const
        {
            return !(*this == other);
        }

        std::string to_string() const;

        /** Semantic accessor for width as x.
         *
         * @return x.
         */
        size_t x() const
        {
            return width;
        }

        /** Semantic accessor for height as y.
         *
         * @return y.
         */
        size_t y() const
        {
            return height;
        }

    public:
        size_t width  = {}; /**< Width of the image region or rectangle */
        size_t height = {}; /**< Height of the image region or rectangle */
    };

} // namespace BatmanInfer

#endif //BATMANINFER_BI_SIZE_2D_H
