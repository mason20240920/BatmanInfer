//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_SHAPE_CALCULATOR_HPP
#define BATMANINFER_BI_SHAPE_CALCULATOR_HPP

#include <data/core/bi_tensor_info.hpp>
#include <data/core/kernel_descriptors.hpp>

namespace BatmanInfer {
    namespace misc {
        namespace shape_calculator {

            /** 计算转置后的矩阵
             *
             * @param input input Input tensor info
             * @return
             */
            inline BITensorShape compute_transposed_shape(const BIITensorInfo &input) {
                BITensorShape shape_transposed{input.tensor_shape()};

                shape_transposed.set(0, input.dimension(1), false);
                shape_transposed.set(1, input.dimension(0), false);

                return shape_transposed;
            }

            /**
             * 计算一个输入张量的交叉形状
             * @param a
             * @param multi_interleave4x4_height
             * @param reinterpret_input_as_3d
             * @return
             */
            inline BITensorShape compute_interleaved_shape(const BIITensorInfo &a,
                                                           int multi_interleave4x4_height = 1,
                                                           bool reinterpret_input_as_3d = false) {
                // 输出的矩阵的形状: [ a_height * W, ceil(a_width / W) ] where W = 4 * multi_interleave4x4_height
                BI_COMPUTE_ERROR_ON(multi_interleave4x4_height < 1);
                // 计算交错宽度 W = 4 * multi_interleave4x4_height
                const int interleave_width = 4 * multi_interleave4x4_height;
                BITensorShape shape_interleaved_a{a.tensor_shape()};
                shape_interleaved_a.set(0, a.dimension(0) * interleave_width);
                if (reinterpret_input_as_3d) {
                    // 将输入张量的第1维和第2维合并为一个新的高度维度M
                    const int M = a.dimension(1) * a.dimension(2);
                    // 计算新的高度维度，按交错宽度进行分块，向上取整
                    const int height = std::ceil(M / static_cast<float>(interleave_width));
                    // 设置1维的高度
                    shape_interleaved_a.set(1, height);

                    // 当数据格式为 NHWC 且形状为 Nx1x1 时，
                    // 张量的维度数可能会自动设置为 1，而不是 3。
                    // 为了避免移除不存在的维度导致的错误，需要检查维度数是否大于 2。
                    if (shape_interleaved_a.num_dimensions() > 2)
                        shape_interleaved_a.remove_dimension(2);
                } else
                    shape_interleaved_a.set(1, std::ceil(a.dimension(1) / static_cast<float>(interleave_width)));

                return shape_interleaved_a;
            }

            inline BITensorShape compute_transpose_1xw_with_element_size_shape(const BIITensorInfo &b,
                                                                               int multi_transpose_1xw_width = 1) {
                // 注意：multi_transpose_1xw_width 表示我们希望将大小为 1x(W) 的块存储在同一行中的数量
                //       transpose1xW 输出矩阵将具有以下形状：
                //       [b_height * W, ceil(b_width / W)] 其中 W = (16 / 张量元素大小) * multi_transpose_1xw_width
                BI_COMPUTE_ERROR_ON(multi_transpose_1xw_width < 1);
                BITensorShape shape_transposed_1xw_b{b.tensor_shape()};
                const size_t transpose_width = (16 / b.element_size()) * multi_transpose_1xw_width;
                shape_transposed_1xw_b.set(BIWindow::DimX, b.dimension(1) * transpose_width);
                shape_transposed_1xw_b.set(BIWindow::DimY, static_cast<size_t>(std::ceil(
                        b.dimension(0) / static_cast<float>(transpose_width))));

                return shape_transposed_1xw_b;
            }

            inline BITensorShape compute_mm_shape(const BIITensorInfo &input0,
                                                  const BIITensorInfo &input1,
                                                  bool is_interleaved_transposed,
                                                  const BIGemmReshapeInfo &reshape_info) {
                BI_COMPUTE_ERROR_ON_MSG(input0.num_dimensions() > 4,
                                        "The number of dimensions for the matrix A must be <= 4");
                BI_COMPUTE_ERROR_ON_MSG(
                        is_interleaved_transposed && reshape_info.reinterpret_input_as_3d(),
                        "The first input tensor cannot be reinterpreted as 3D if is_interleaved_transposed is true");

                const bool reinterpret_input_as_3d = reshape_info.reinterpret_input_as_3d();
                const bool reinterpret_output_as_3d = reshape_info.depth_output_gemm3d() != 0;
                const int depth_output_gemm3d = reinterpret_output_as_3d ? reshape_info.depth_output_gemm3d() : 1;
                const int m =
                        reshape_info.reinterpret_input_as_3d() ? input0.dimension(1) * input0.dimension(2)
                                                               : input0.dimension(1);

                // If the output of GEMM has to be reinterpreted as 3D, the number of input0 rows (M) is obtained collapsing the second and third
                // dimension of the output tensor
                const int dim0 = is_interleaved_transposed ? reshape_info.n() : input1.dimension(0);
                const int dim1 = is_interleaved_transposed ? reshape_info.m() / depth_output_gemm3d : m /
                                                                                                      depth_output_gemm3d;
                const int dim2 = reinterpret_input_as_3d ? input0.tensor_shape()[3] : input0.tensor_shape()[2];
                const int dim3 = reinterpret_input_as_3d ? 1 : input0.tensor_shape()[3];

                BITensorShape output_shape{input0.tensor_shape()};

                output_shape.set(0, dim0);
                output_shape.set(1, dim1);
                output_shape.set(2, reinterpret_output_as_3d ? depth_output_gemm3d : dim2);
                output_shape.set(3, reinterpret_output_as_3d ? dim2 : dim3);
                output_shape.set(4, reinterpret_output_as_3d ? dim3 : 1);

                return output_shape;
            }

/** Calculate the matrix multiplication output shape of two tensors
 *
 * @param[in] input0    First input tensor info
 * @param[in] input1    Second input tensor info
 * @param[in] gemm_info GEMM reshape info
 *
 * @return the calculated shape
 */
            inline BITensorShape
            compute_mm_shape(const BIITensorInfo &input0,
                             const BIITensorInfo &input1,
                             const BIGemmReshapeInfo &gemm_info) {
                BI_COMPUTE_UNUSED(input1);
                BI_COMPUTE_ERROR_ON_MSG(input0.num_dimensions() > 4,
                                        "The number of dimensions for the matrix A must be <= 4");

                const bool reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
                const bool reinterpret_output_as_3d = gemm_info.depth_output_gemm3d() != 0;
                const int depth_output_gemm3d = reinterpret_output_as_3d ? gemm_info.depth_output_gemm3d() : 1;

                BITensorShape output_shape{input0.tensor_shape()};

                if (!reinterpret_input_as_3d && !reinterpret_output_as_3d) {
                    output_shape.set(0, gemm_info.n());
                    output_shape.set(1, gemm_info.m());
                } else {
                    // If the output of GEMM has to be reinterpreted as 3D, the number of input0 rows (M) is obtained collapsing the second and third
                    // dimension of the output tensor
                    const int batch_size = reinterpret_input_as_3d ? input0.tensor_shape()[3]
                                                                   : input0.tensor_shape()[2];
                    output_shape.set(0, gemm_info.n());
                    output_shape.set(1, gemm_info.m() / depth_output_gemm3d);
                    output_shape.set(2, reinterpret_output_as_3d ? depth_output_gemm3d : batch_size);
                    output_shape.set(3, reinterpret_output_as_3d ? batch_size : 1);
                }

                return output_shape;
            }


            /**
             * 计算两个张量矩阵乘法的输出形状
             * @param input0
             * @param input1
             * @param gemm_info
             * @return
             */
            inline BITensorShape
            compute_mm_shape(const BIITensorInfo &input0,
                             const BIITensorInfo &input1,
                             const GEMMKernelInfo &gemm_info) {
                BI_COMPUTE_UNUSED(input1);
                BI_COMPUTE_ERROR_ON_MSG(input0.num_dimensions() > 4,
                                        "The number of dimensions for the matrix A must be <= 4");

                const bool reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d;
                const bool reinterpret_output_as_3d = gemm_info.depth_output_gemm3d != 0;
                const unsigned int depth_output_gemm3d = reinterpret_output_as_3d ? gemm_info.depth_output_gemm3d : 1;

                BITensorShape output_shape{input0.tensor_shape()};

                if (!reinterpret_input_as_3d && !reinterpret_output_as_3d) {
                    output_shape.set(0, gemm_info.n);
                    output_shape.set(1, gemm_info.m);
                } else {
                    // If the output of GEMM has to be reinterpreted as 3D, the number of input0 rows (M) is obtained collapsing the second and third
                    // dimension of the output tensor
                    const unsigned int batch_size = reinterpret_input_as_3d ? input0.tensor_shape()[3]
                                                                            : input0.tensor_shape()[2];
                    output_shape.set(0, gemm_info.n);
                    output_shape.set(1, gemm_info.m / depth_output_gemm3d);
                    output_shape.set(2, reinterpret_output_as_3d ? depth_output_gemm3d : batch_size);
                    output_shape.set(3, reinterpret_output_as_3d ? batch_size : 1);
                }

                return output_shape;
            }

            /** Get the tensor shape
             *
             * @param[in] data Input data
             *
             * @return the extracted tensor shape
             */
            template <typename T>
            inline BITensorShape extract_shape(T *data)
            {
                return data->info()->tensor_shape();
            }

            inline BITensorShape extract_shape(BIITensorInfo *data)
            {
                return data->tensor_shape();
            }

            inline BITensorShape extract_shape(const BIITensorInfo *data)
            {
                return data->tensor_shape();
            }

            inline BITensorShape extract_shape(const BITensorShape *data)
            {
                return *data;
            }

            inline BITensorShape extract_shape(BITensorShape *data)
            {
                return *data;
            }

            /** Calculate the reduced shape of a tensor given an axis
             *
             * @param[in] input     Input tensor info
             * @param[in] axis      Axis on which to perform reduction
             * @param[in] keep_dims (Optional) Whether to keep the dimension after reduction operation. Defaults to true.
             *
             * @return the calculated shape
             */
            inline BITensorShape compute_reduced_shape(const BITensorShape &input, unsigned int axis, bool keep_dims = true)
            {
                BITensorShape output_shape{input};

                if (!keep_dims)
                {
                    output_shape.remove_dimension(axis);
                }
                else
                {
                    output_shape.set(axis, 1);
                }

                return output_shape;
            }

            /** Calculate the concatenate output shape of the concatenate operation along a single axis
             *
             * @param[in] input Vector containing the shapes of the inputs
             * @param[in] axis  Axis along which to concatenate the input tensors
             *
             * @return the calculated shape
             */
            template <typename T>
            inline BITensorShape calculate_concatenate_shape(const std::vector<T *> &input, size_t axis)
            {
                BITensorShape out_shape = extract_shape(input[0]);

#if defined(BI_COMPUTE_ASSERTS_ENABLED)
                // All dimensions must match except the axis one
                for (unsigned int i = 0; i < MAX_DIMS; ++i)
                {
                    if (i == axis)
                    {
                        continue;
                    }

                    for (const auto &tensor : input)
                    {
                        BI_COMPUTE_ERROR_ON(tensor == nullptr);
                        const BITensorShape shape = extract_shape(tensor);
                        BI_COMPUTE_ERROR_ON(out_shape[i] != shape[i]);
                    }
                }
#endif // defined(BI_COMPUTE_ASSERTS_ENABLED)

                // Calculate output shape
                size_t new_size = 0;
                for (const auto &tensor : input)
                {
                    const BITensorShape shape = extract_shape(tensor);
                    new_size += shape[axis];
                }

                out_shape.set(axis, new_size);

                return out_shape;
            }

        }
    }
}

#endif //BATMANINFER_BI_SHAPE_CALCULATOR_HPP
