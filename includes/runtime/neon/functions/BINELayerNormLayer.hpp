//
// Created by Mason on 2025/8/2.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <runtime/bi_tensor.hpp>

#include <memory>

namespace BatmanInfer {
    class BIITensor;

    namespace cpu {
        class BINELayerNormLayerKernel;
    }

    /** Basic function to compute a RMS normalization layer. This function calls the following kernels:
     *
     * -# @ref BINERMSNormLayerKernel
     *
     */
    class BINELayerNormLayer final : public BIIFunction {
    public:
        /** Default constructor */
        BINELayerNormLayer();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINELayerNormLayer(const BINELayerNormLayer &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINELayerNormLayer &operator=(const BINELayerNormLayer &) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINELayerNormLayer(BINELayerNormLayer &&) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINELayerNormLayer &operator=(BINELayerNormLayer &&) = delete;

        /** Default destructor */
        ~BINELayerNormLayer();

        /** Set the input and output tensors.
        *
        * Valid data layouts:
        * - [H,S] = [hidden size, sequence length]
        *
        * Valid data type configurations:
        * |src      |dst       |
        * |:--------|:---------|
        * |F16      |F16       |
        *
        * @param[in]  input     Source tensor. 2 dimensions : F16. Data layouts supported: HS.
         * @param[in] gamma
        * @param[out] output    Destination with the same dimensions, data type, data layout and number of channels of  @p input
        */
        void configure(const BIITensor *input, const BIITensor *gamma, const BIITensor *beta, BIITensor *output);

        void dynamic_configure(const BIITensor *input) const;

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *gamma,
                                 const BIITensor *beta,
                                 const BIITensorInfo *output);

        void run() override;

    private:
        std::unique_ptr<cpu::BINELayerNormLayerKernel> _layer_norm_kernel; /**<Layer Normalization layer kernel */
        BITensor _input;
        BITensor _gamma;
        BITensor _beta;
    };
}