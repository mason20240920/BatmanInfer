//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_BI_SIZE_3D_H
#define BATMANINFER_BI_SIZE_3D_H

#include <string>

namespace BatmanInfer {

    /** Class for specifying the size of a 3D shape or object */
    class Size3D
    {
    public:
        /** Default constructor */
        Size3D() = default;
        /** Constructor. Initializes "width", "height" and "depth" respectively with "w", "h" and "d"
         *
         * @param[in] w Width of the 3D shape or object
         * @param[in] h Height of the 3D shape or object
         * @param[in] d Depth of the 3D shape or object
         */
        Size3D(size_t w, size_t h, size_t d) noexcept : width(w), height(h), depth(d)
        {
        }

        /** Convert the values stored to string
         *
         * @return string of (width x height x depth).
         */
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

        /** Semantic accessor for depth as z.
         *
         * @return z.
         */
        size_t z() const
        {
            return depth;
        }

        bool operator!=(const Size3D &other) const
        {
            return !(*this == other);
        }

        bool operator==(const Size3D &other) const
        {
            return (width == other.width) && (height == other.height) && (depth == other.depth);
        }

    public:
        size_t width  = {}; /**< Width of the 3D shape or object */
        size_t height = {}; /**< Height of the 3D shape or object */
        size_t depth  = {}; /**< Depth of the 3D shape or object */
    };

} // namespace BatmanInfer

#endif //BATMANINFER_BI_SIZE_3D_H
