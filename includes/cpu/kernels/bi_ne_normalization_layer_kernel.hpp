//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <data/core/neon/bi_i_ne_kernel.hpp>

namespace BatmanInfer {
    class BIITensor;

    class BINENormalizationLayerKernel : public BIINEKernel {
    public:
        const char *name() const override {
            return "BINENormalizationLayerKernel";
        }

        /** Default constructor */
        BINENormalizationLayerKernel();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINENormalizationLayerKernel(const BINENormalizationLayerKernel &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINENormalizationLayerKernel &operator=(const BINENormalizationLayerKernel &) = delete;

        /** Default Move Constructor. */
        BINENormalizationLayerKernel(BINENormalizationLayerKernel
                                     &&) = default;

        /** Default move assignment operator */
        BINENormalizationLayerKernel &operator=(BINENormalizationLayerKernel &&) = default;

        /** Default destructor */
        ~BINENormalizationLayerKernel() = default;

        /** Set the input and output tensors.
         *
         * @param[in]  input         Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
         *                           and an optional 4th dimension for batch of inputs. Data types supported: FP16/F32. Data layouts supported: NCHW/NHWC.
         * @param[in]  input_squared Source with each element has been squared. 3 lower dims represent a single input with dimensions [width, height, IFM],
         *                           Data type and layout supported: same as @p input.
         * @param[out] output        Destination tensor. Output will have the same number of dimensions as input. Data type and layout supported: same as @p input.
         * @param[in]  norm_info     Normalization layer information like the normalization type, normalization size and other parameters.
         */
        void
        configure(const BIITensor *input, const BIITensor *input_squared, BIITensor *output,
                  BINormalizationLayerInfo norm_info);

        /** Static function to check if given info will lead to a valid configuration of @ref NENormalizationLayerKernel
         *
         * @param[in] input         Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
         *                          and an optional 4th dimension for batch of inputs. Data types supported: FP16/F32. Data layouts supported: NCHW/NHWC.
         * @param[in] input_squared Source with each element has been squared. 3 lower dims represent a single input with dimensions [width, height, IFM],
         *                          Data type and layout supported: same as @p input.
         * @param[in] output        Destination tensor. Output will have the same number of dimensions as input. Data type and layout supported: same as @p input.
         * @param[in] norm_info     Normalization layer information like the normalization type, normalization size and other parameters.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *input_squared,
                                 const BIITensorInfo *output,
                                 BINormalizationLayerInfo norm_info);

        // Inherited methods overridden:
        void run(const BIWindow &window, const ThreadInfo &info) override;

    private:
        /** Common signature for all the specialised normalization functions
         *
         * @param[in] window Region on which to execute the kernel.
         */
        using NormalizationFunction = void (*)(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

    private:
        NormalizationFunction _func;
        const BIITensor *_input;
        const BIITensor *_input_squared;
        BIITensor *_output;
        BINormalizationLayerInfo _norm_info;
    };
}; // namespace BatmanInfer