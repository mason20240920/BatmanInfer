//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Basic function to execute flatten layer kernel. */
    class BINEFlattenLayer : public BIIFunction
    {
    public:
        BINEFlattenLayer();
        /** Destructor */
        ~BINEFlattenLayer();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEFlattenLayer(const BINEFlattenLayer &) = delete;
        /** Default move constructor */
        BINEFlattenLayer(BINEFlattenLayer &&);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEFlattenLayer &operator=(const BINEFlattenLayer &) = delete;
        /** Default move assignment operator */
        BINEFlattenLayer &operator=(BINEFlattenLayer &&);
        /** Initialise the kernel's input and output.
         *
         *  Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |All            |All            |
         *
         * @param[in]  input  First input tensor to flatten with at least 3 dimensions. The dimensions over the third will be interpreted as batches. Data types supported: All
         * @param[out] output Output tensor with shape [w*h*d, input_batches] where:
         *             w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
         */
        void configure(const BIITensor *input, BIITensor *output);

        /** Static function to check if given info will lead to a valid configuration of @ref NEFlattenLayer
         *
         * @param[in]  input  First input tensor to flatten with at least 3 dimensions.
         *                    The dimensions above the third will be interpreted as batches. Data types supported: All
         * @param[out] output Output tensor with shape [w*h*d, input_batches] where:
         *                    w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
