//
// Created by Mason on 2025/1/22.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>

namespace BatmanInfer {
    // 前向生命
    class BIITensor;

    class BIITensorInfo;

    class BINEConvertFullyConnectedWeights : public BIIFunction {
    public:
        /** Default constructor */
        BINEConvertFullyConnectedWeights();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEConvertFullyConnectedWeights(const BINEConvertFullyConnectedWeights &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEConvertFullyConnectedWeights &operator=(const BINEConvertFullyConnectedWeights &) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEConvertFullyConnectedWeights(BINEConvertFullyConnectedWeights
                                         &&) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEConvertFullyConnectedWeights &operator=(BINEConvertFullyConnectedWeights &&) = delete;

        /** Default destructor */
        ~BINEConvertFullyConnectedWeights();

        /** Initialize the function.
         *
         * Valid data layouts:
         * - NHWC
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |All            |All            |
         *
         * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: All.
         * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
         * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
         * @param[in]  data_layout          The data layout the weights have been trained in.
         */
        void
        configure(const BIITensor *input, BIITensor *output, const BITensorShape &original_input_shape,
                  BIDataLayout data_layout);

        /** Static function to check if given info will lead to a valid configuration of @ref NEConvertFullyConnectedWeights
         *
         * @param[in] input                Source weights tensor info to convert. Must be 2 dimensional. Data types supported: All.
         * @param[in] output               The converted weights tensor info. Shape and Data Type: Same as @p input.
         * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
         * @param[in] data_layout          The data layout the weights have been trained in.
         *
         * @return A Status
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *output,
                                 const BITensorShape &original_input_shape,
                                 BIDataLayout data_layout);

        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}