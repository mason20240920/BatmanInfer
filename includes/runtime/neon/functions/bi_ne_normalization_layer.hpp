//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/neon/functions/ne_pixel_wise_multiplication.hpp>
#include <runtime/bi_tensor.hpp>

#include <memory>

namespace BatmanInfer {
    class BIITensor;

    class BINENormalizationLayerKernel;

    /** Basic function to compute a normalization layer. This function calls the following kernels:
     *
     * -# @ref BINEPixelWiseMultiplication
     * -# NEFillBorderKernel
     * -# NENormalizationLayerKernel
     *
     */
    class BINENormalizationLayer : public BIIFunction {
    public:
        /** Default constructor */
        BINENormalizationLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINENormalizationLayer() : BINENormalizationLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINENormalizationLayer(const BINENormalizationLayer &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINENormalizationLayer &operator=(const BINENormalizationLayer &) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINENormalizationLayer(BINENormalizationLayer &&) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINENormalizationLayer &operator=(BINENormalizationLayer &&) = delete;

        /** Default destructor */
        ~BINENormalizationLayer();

        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src      |dst       |
         * |:--------|:---------|
         * |F32      |F32       |
         * |F16      |F16       |
         *
         * @param[in]  input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
         *                       and an optional 4th dimension for batch of inputs. Data type supported: F16/F32. Data layouts supported: NCHW/NHWC.
         * @param[out] output    Destination with the same dimensions, data type, data layout and number of channels of  @p input
         * @param[in]  norm_info Normalization layer information like the normalization type, normalization size and other parameters.
         */
        void configure(const BIITensor *input, BIITensor *output, const BINormalizationLayerInfo &norm_info);

        /** Static function to check if given info will lead to a valid configuration of @ref NENormalizationLayer
         *
         * @param[in] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
         *                      and an optional 4th dimension for batch of inputs. Data type supported: F16/F32. Data layouts supported: NCHW/NHWC.
         * @param[in] output    Destination with the same dimensions, data type, data layout and number of channels of  @p input
         * @param[in] norm_info Normalization layer information like the normalization type, normalization size and other parameters.
         *
         * @return a status
         */
        static BIStatus
        validate(const BIITensorInfo *input, const BIITensorInfo *output, const BINormalizationLayerInfo &norm_info);

        // Inherited methods overridden:
        void run() override;

    private:
        BIMemoryGroup _memory_group; /**< Function memory group */
        std::unique_ptr<BINENormalizationLayerKernel> _norm_kernel;  /**< Normalization layer kernel */
        BINEPixelWiseMultiplication _multiply_f;   /**< Pixel multiplication function */
        BITensor _input_squared; /**< The intermediate buffer which stores results of squaring input */
    };
} // namespace BatmanInfer