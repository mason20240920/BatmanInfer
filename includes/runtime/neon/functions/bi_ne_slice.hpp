//
// Created by Mason on 2025/1/15.
//

#pragma once

#include <runtime/bi_i_function.hpp>
#include <runtime/neon/bi_i_ne_operator.hpp>
#include <data/core/bi_coordinates.hpp>
#include <data/core/bi_i_tensor_info.hpp>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    /**
     * 用于执行张量切片的基函数
     */
    class BINESlice : public BIIFunction {
    public:
        /** Default Constructor */
        BINESlice();

        /** Default Destructor */
        ~BINESlice();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINESlice(const BINESlice &) = delete;

        /** Default move constructor */
        BINESlice(BINESlice &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINESlice &operator=(const BINESlice &) = delete;

        /** Default move assignment operator */
        BINESlice &operator=(BINESlice &&);

        /** Configure kernel
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src    |dst    |
         * |:------|:------|
         * |All    |All    |
         *
         * @note Supported tensor rank: up to 4
         * @note Start indices must be non-negative. 0 <= starts[i]
         * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
         * @note End indices are not inclusive unless negative.
         *
         * @param[in]  input  Source tensor. Data type supported: All
         * @param[out] output Destination tensor. Data type supported: Same as @p input
         * @param[in]  starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in]  ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         */
        void
        configure(const BIITensor *input, BIITensor *output, const BICoordinates &starts, const BICoordinates &ends);

        /** Static function to check if given info will lead to a valid configuration of @ref NESlice
         *
         * @note Supported tensor rank: up to 4
         * @note Start indices must be non-negative. 0 <= starts[i]
         * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
         * @note End indices are not inclusive unless negative.
         *
         * @param[in] input  Source tensor info. Data type supported: All
         * @param[in] output Destination tensor info. Data type supported: Same as @p input
         * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         *
         * @return A status
         */
        static BIStatus
        validate(const BIITensorInfo *input, const BIITensorInfo *output, const BICoordinates &starts,
                 const BICoordinates &ends);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

    namespace experimental {
        /** Basic function to perform tensor slicing */
        class BINESlice : public BIINEOperator {
        public:
            /** Configure kernel
             *
             * @note Supported tensor rank: up to 4
             * @note Start indices must be non-negative. 0 <= starts[i]
             * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
             * @note End indices are not inclusive unless negative.
             *
             * @param[in]  input  Source tensor info. Data type supported: All
             * @param[out] output Destination tensor info. Data type supported: Same as @p input
             * @param[in]  starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
             * @param[in]  ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
             */
            void configure(const BIITensorInfo *input, BIITensorInfo *output, const BICoordinates &starts,
                           const BICoordinates &ends);

            /** Static function to check if given info will lead to a valid configuration of @ref NESlice
             *
             * @note Supported tensor rank: up to 4
             * @note Start indices must be non-negative. 0 <= starts[i]
             * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
             * @note End indices are not inclusive unless negative.
             *
             * @param[in] input  Source tensor info. Data type supported: All
             * @param[in] output Destination tensor info. Data type supported: Same as @p input
             * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
             * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
             *
             * @return A status
             */
            static BIStatus
            validate(const BIITensorInfo *input, const BIITensorInfo *output, const BICoordinates &starts,
                     const BICoordinates &ends);
        };
    }
}