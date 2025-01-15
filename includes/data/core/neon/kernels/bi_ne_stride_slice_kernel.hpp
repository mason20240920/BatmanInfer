//
// Created by Mason on 2025/1/15.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <data/core/neon/bi_i_ne_kernel.hpp>

#include <cstdint>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    /**
     * 内核声明: 用于内核执行张量步长切片的接口
     */
    class BINEStridedSliceKernel : public BIINEKernel {
    public:
        const char *name() const override {
            return "BINEStridedSliceKernel";
        }

        BINEStridedSliceKernel();

        BINEStridedSliceKernel(const BINEStridedSliceKernel &) = delete;

        BINEStridedSliceKernel &operator=(const BINEStridedSliceKernel &) = delete;

        BINEStridedSliceKernel(BINEStridedSliceKernel &&) = default;

        BINEStridedSliceKernel &operator=(BINEStridedSliceKernel &&) = default;

        ~BINEStridedSliceKernel() override = default;

        /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input            Source tensor info. Data type supported: All
     * @param[out] output           Destination tensor info. Data type supported: Same as @p input
     * @param[in]  starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  begin_mask       If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  end_mask         If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  shrink_axis_mask If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
        void configure(const BIITensorInfo *input,
                       BIITensorInfo *output,
                       const BICoordinates &starts,
                       const BICoordinates &ends,
                       const BiStrides &strides,
                       int32_t begin_mask,
                       int32_t end_mask,
                       int32_t shrink_axis_mask);

        /** Static function to check if given info will lead to a valid configuration of @ref NEStridedSliceKernel
         *
         * @note Supported tensor rank: up to 4
         *
         * @param[in] input            Source tensor info. Data type supported: All
         * @param[in] output           Destination tensor info. Data type supported: Same as @p input
         * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] begin_mask       If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
         * @param[in] end_mask         If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
         * @param[in] shrink_axis_mask If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
         *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *output,
                                 const BICoordinates &starts,
                                 const BICoordinates &ends,
                                 const BiStrides &strides,
                                 int32_t begin_mask,
                                 int32_t end_mask,
                                 int32_t shrink_axis_mask);

        // Inherited methods overridden:
        void run_op(BIITensorPack &tensors,
                    const BIWindow &window,
                    const ThreadInfo &info) override;

    private:
        // 大概开始的坐标
        BICoordinates _starts_abs;
        // 结束的步长
        BICoordinates _final_strides;
        // 收缩轴掩码
        int32_t _shrink_mask;
    };
}