//
// Created by Mason on 2025/4/1.
//

#pragma once

#include <data/core/bi_types.hpp>

#include <data/core/neon/bi_i_ne_kernel.hpp>

namespace BatmanInfer {
    // 前向生命
    class BIITensor;

    class BINEGatherKernel : public BIINEKernel {
    public:
        /** Default constructor. */
        BINEGatherKernel();

        /** Prevent instances of this class from being copied (As this class contains pointers). */
        BINEGatherKernel(const BINEGatherKernel &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers). */
        BINEGatherKernel &operator=(const BINEGatherKernel &) = delete;

        /** Allow instances of this class to be moved. */
        BINEGatherKernel(BINEGatherKernel &&) = default;

        /** Allow instances of this class to be moved. */
        BINEGatherKernel &operator=(BINEGatherKernel &&) = default;

        /** Name of the kernel
        *
        * @return Kernel name
        */
        const char *name() const override {
            return "BINEGatherKernel";
        }

        /** Initialise the kernel's inputs and outputs
        *
        * @param[in]  input   Source tensor. Supported tensor rank: up to 4. Data type supported: All
        * @param[in]  indices Indices tensor. Supported tensor rank: up to 3. Must be one of the following type: U32/S32. Each value Must be in range [0, input.shape[@p axis])
        *                     @note 2D or 3D indices are only supported for the axis 1.
        * @param[out] output  Destination tensor. Data type supported: Same as @p input
        * @param[in]  axis    (Optional) The axis in @p input to gather @p indices from. Negative values wrap around. Defaults to 0.
        *
        */
        void configure(const BIITensor *input, const BIITensor *indices, BIITensor *output, int axis = 0);

        /** Static function to check if given info will lead to a valid configuration
         *
         * Similar to @ref BINEGatherKernel::configure()
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *indices,
                                 const BIITensorInfo *output,
                                 int axis);

        /**
         * @brief 动态初始化的configure, 配置需要修改的参数, 其余参数
         * @param input
         * @param output
         */
        void dynamic_configure(const BIITensor *indices, BIITensor *output);

        /**
         * @brief
         * @param window
         * @param info
         */
        void run(const BIWindow &window, const ThreadInfo &info) override;

    private:
        template<typename TIndex>
        void gather_common(const BIWindow &window, const ThreadInfo &info);

        using kernel_ptr = void (BINEGatherKernel::*)(
            const BIWindow &window, const ThreadInfo &info);

        const BIITensor *_input;
        const BIITensor *_indices;
        int _axis;
        BIITensor *_output;
        kernel_ptr _func;

        BIStrides _src_it_strides;
        BIStrides _idx_it_strides;
    };
}
