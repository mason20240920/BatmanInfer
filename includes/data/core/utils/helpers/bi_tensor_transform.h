//
// Created by holynova on 2025/1/14.
//

#pragma once

#include "data/core/bi_types.hpp"

namespace BatmanInfer {

    namespace helpers {

        namespace tensor_transform {

            /** Computes stride of a given index
             *
             * @param[in] index   Index of tensor to calculate absolute start position
             * @param[in] strides Slice strides
             *
             * @return Stride at a given index
             */
            int calculate_stride_on_index(int index, BICoordinates strides);

            /** Computes absolute start position of a given index for a strided slice operation
             *
             * @param[in] input_shape Input tensor shape
             * @param[in] index       Index of tensor to calculate absolute start position
             * @param[in] starts      Start coordinates
             * @param[in] strides     Slice strides
             * @param[in] begin_mask  (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and
             *                        the fullest possible range in that dimension is used instead.
             *
             * @return Absolute start position of a given index
             */
            int calculate_start_on_index(
                    BITensorShape input_shape, int index, BICoordinates starts, BICoordinates strides,
                    int32_t begin_mask);

            /** Returns the absolute end position of a given index for a strided slice operation
             *
             * @param[in] input_shape      Input tensor shape
             * @param[in] index            Index of tensor to calculate absolute start position
             * @param[in] start_on_index   Absolute start coordinate for given index
             * @param[in] ends             End coordinates
             * @param[in] strides          Slice strides
             * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, end[i] is ignored and
             *                             the fullest possible range in that dimension is used instead.
             * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
             *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
             *
             * @return Absolute end position of a given index
             */
            int calculate_end_on_index(BITensorShape input_shape,
                                       int index,
                                       int start_on_index,
                                       BICoordinates ends,
                                       BICoordinates strides,
                                       int32_t end_mask = 0,
                                       int32_t shrink_axis_mask = 0);

            /** Calculate start, end and stride coordinates for a strided slice
             *
             * @param[in] input_shape      Input tensor shape
             * @param[in] starts           Start coordinates
             * @param[in] ends             End coordinates
             * @param[in] strides          Slice strides
             * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and
             *                             the fullest possible range in that dimension is used instead.
             * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, end[i] is ignored and
             *                             the fullest possible range in that dimension is used instead.
             * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
             *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
             *
             * @return A tuple with <Start,End,Strides>
             */
            std::tuple<BICoordinates, BICoordinates, BICoordinates> calculate_strided_slice_coords(
                    BITensorShape input_shape,
                    BICoordinates starts,
                    BICoordinates ends,
                    BICoordinates strides,
                    int32_t begin_mask = 0,
                    int32_t end_mask = 0,
                    int32_t shrink_axis_mask = 0);

            /** Computes output shape of strided slice
             *
             * @warning Starts and ends must be non-negative
             * @warning Starts, ends and final strides should have the same dimensions as the input shape
             *
             * @param[in] input_shape       Input tensor shape
             * @param[in] starts            Absolute start coordinates
             * @param[in] ends              Absolute end coordinates
             * @param[in] strides           Slice strides
             * @param[in] begin_mask        (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and
             *                              the fullest possible range in that dimension is used instead.
             * @param[in] end_mask          (Optional) If the ith bit of end_mask is set, end[i] is ignored and
             *                              the fullest possible range in that dimension is used instead.
             * @param[in] shrink_axis_mask  (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
             *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
             * @param[in] return_unshrinked (Optional) Returns un-shrinked shape
             *
             * @return The output tensor shape
             */
            BITensorShape compute_strided_slice_output_shape(BITensorShape input_shape,
                                                             BICoordinates starts,
                                                             BICoordinates ends,
                                                             BICoordinates strides,
                                                             int32_t begin_mask = 0,
                                                             int32_t end_mask = 0,
                                                             int32_t shrink_axis_mask = 0,
                                                             bool return_unshrinked = false);

            /** Constructs end mask in case we want to perform a slice operation using the strided slice interface
             *
             * @note Ends are inclusive in slice operations that is why construction an end mask is needed
             *
             * @param[in] ends End coordinates
             *
             * @return End mask
             */
            int32_t construct_slice_end_mask(BICoordinates ends);

        } // namespace tensor_transform

    } // namespace helpers

} // namespace BatmanInfer
