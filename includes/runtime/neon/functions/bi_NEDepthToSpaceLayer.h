//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

    // Forward declarations
    class BIITensor;
    class BIITensorInfo;
    class BINEDepthToSpaceLayerKernel;

    /** Basic function to run NEDepthToSpaceLayerKernel. */
    class BINEDepthToSpaceLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEDepthToSpaceLayer();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDepthToSpaceLayer(const BINEDepthToSpaceLayer &) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDepthToSpaceLayer &operator=(const BINEDepthToSpaceLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEDepthToSpaceLayer(BINEDepthToSpaceLayer &&) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEDepthToSpaceLayer &operator=(BINEDepthToSpaceLayer &&) = delete;
        /** Default destructor */
        ~BINEDepthToSpaceLayer();
        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |All            |All            |
         *
         * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All
         * @param[out] output      Tensor output. Data types supported: same as @p input
         * @param[in]  block_shape Block shape value.
         */
        void configure(const BIITensor *input, BIITensor *output, int32_t block_shape);
        /** Static function to check if given info will lead to a valid configuration of @ref NEDepthToSpaceLayer.
         *
         * @param[in] input       Tensor input info. Supported tensor rank: 4. Data types supported: All
         * @param[in] output      Tensor output info. Data types supported: same as @p input
         * @param[in] block_shape Block shape x value.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output, int32_t block_shape);

        void run() override;

    private:
        std::unique_ptr<BINEDepthToSpaceLayerKernel> _kernel;
    };

} // namespace BatmanInfer
