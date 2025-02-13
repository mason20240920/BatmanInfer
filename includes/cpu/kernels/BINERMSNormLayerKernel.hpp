//
// Created by Mason on 2025/2/12.
//

#pragma once

#include <data/core/neon/bi_i_ne_kernel.hpp>

namespace BatmanInfer {
    class BIITensor;

    class BINERMSNormLayerKernel : public BIINEKernel {
    public:
        const char *name() const override {
            return "BINERMSNormLayerKernel";
        }

        BINERMSNormLayerKernel();

        BINERMSNormLayerKernel(const BINERMSNormLayerKernel &) = delete;

        BINERMSNormLayerKernel &operator=(const BINERMSNormLayerKernel &) = delete;

        BINERMSNormLayerKernel(BINERMSNormLayerKernel &&) = default;

        BINERMSNormLayerKernel &operator=(BINERMSNormLayerKernel &&) = default;

        ~BINERMSNormLayerKernel() = default;

        bool is_parallelisable() const override;

        /** Set the input and output tensors.
         *
         * @param[in]  input         Source tensor. 2 lower dims represent a single input with dimensions [hidden size ,sequence size]
         * @param[in]  gamma
         * @param[out] output        Destination tensor. Output will have the same number of dimensions as input. Data type and layout supported: same as @p input.
         * @param[in]  norm_info     Normalization layer information like the normalization type, normalization size and other parameters.
         */
        void
        configure(const BIITensor *input, const BIITensor *gamma, BIITensor *output);

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *gamma,
                                 const BIITensorInfo *output);

        // Inherited methods overridden:
        void run(const BIWindow &window, const ThreadInfo &info) override;

    private:
        /** Common signature for all the specialised normalization functions
        *
        * @param[in] window Region on which to execute the kernel.
        */
        using RMSNormalizationFunction = void (*)(const BIWindow &win, const BIITensor *in, const BIITensor *gamma,
                                                  BIITensor *out);

    private:
        RMSNormalizationFunction _func;
        const BIITensor *_input;
        const BIITensor *_gamma;
        BIITensor *_output;
    };
}
