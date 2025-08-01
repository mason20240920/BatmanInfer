//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/kernel_descriptors.hpp"
#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Basic function to compute Scale */
    class BINEScale : public BIIFunction
    {
    public:
        /** Constructor */
        BINEScale();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEScale(const BINEScale &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEScale(BINEScale &&) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEScale &operator=(const BINEScale &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEScale &operator=(BINEScale &&) = delete;
        /** Destructor */
        ~BINEScale();
        /** Initialize the function's source, destination, interpolation type and border_mode.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |QASYMM8        |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |
         * |F16            |F16            |
         * |F32            |F32            |
         * |U8             |U8             |
         * |S8             |S8             |
         * |S16            |S16            |
         *
         * @param[in, out] input  Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/U8/S8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
         * @param[out]     output Destination tensor. Data type supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
         * @param[in]      info   @ref ScaleKernelInfo to be used for configuration
         *
         * @note Using S8 data type only supports NHWC, @p border_mode Replicate, and @p policy Bilinear
         */
        void configure(BIITensor *input, BIITensor *output, const ScaleKernelInfo &info);
        /** Static function to check if given info will lead to a valid configuration of @ref NEScale
         *
         * @param[in] input  Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/U8/S8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
         * @param[in] output Destination tensor. Data type supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
         * @param[in] info   @ref ScaleKernelInfo to be used for validation
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output, const ScaleKernelInfo &info);

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
