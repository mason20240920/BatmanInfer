//
// Created by Mason on 2025/8/2.
//

#pragma once

#include <data/core/neon/bi_i_ne_kernel.hpp>

namespace BatmanInfer {
    class BIITensor;

    namespace cpu {
        class BINELayerNormLayerKernel : public BIINEKernel {
        public:
            const char *name() const override {
                return "BINELayerNormLayerKernel";
            }

            BINELayerNormLayerKernel();

            BINELayerNormLayerKernel(const BINELayerNormLayerKernel &) = delete;

            BINELayerNormLayerKernel &operator=(const BINELayerNormLayerKernel &) = delete;

            BINELayerNormLayerKernel(BINELayerNormLayerKernel &&) = default;

            BINELayerNormLayerKernel &operator=(BINELayerNormLayerKernel &&) = default;

            ~BINELayerNormLayerKernel() override = default;

            [[nodiscard]] bool is_parallelisable() const override;

            /** Set the input and output tensors.
             *
             * @param[in]  input         Source tensor. 2 lower dims represent a single input with dimensions [hidden size ,sequence size]
             * @param[in]  gamma
             * @param[out] output        Destination tensor. Output will have the same number of dimensions as input. Data type and layout supported: same as @p input.
             */
            void configure(const BIITensor *input,
                           const BIITensor *gamma,
                           const BIITensor *beta,
                           BIITensor *output);

            /**
             * @brief 动态配置输入张量修改window
             * @param input
             */
            void dynamic_configure(const BIITensor *input);

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
            using LayerNormalizationFunction = void (*)(const BIWindow &win,
                                                        const BIITensor *in,
                                                        BIITensor *out,
                                                        const BIITensor *gamma,
                                                        const BIITensor *beta,
                                                        float epsilon);

        private:
            LayerNormalizationFunction _func;
            const BIITensor *_input;
            const BIITensor *_gamma;
            const BIITensor *_beta;
            float _epsilon;
            BIITensor *_output;
        };
    }
}
