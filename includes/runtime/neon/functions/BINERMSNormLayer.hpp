//
// Created by Mason on 2025/2/12.
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
        class BINERMSNormLayerKernel;
    }

    /** Basic function to compute a RMS normalization layer. This function calls the following kernels:
     *
     * -# @ref BINERMSNormLayerKernel
     *
     */
    class BINERMSNormLayer : public BIIFunction {
    public:
        /** Default constructor */
        BINERMSNormLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINERMSNormLayer() : BINERMSNormLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINERMSNormLayer(const BINERMSNormLayer &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINERMSNormLayer &operator=(const BINERMSNormLayer &) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINERMSNormLayer(BINERMSNormLayer &&) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINERMSNormLayer &operator=(BINERMSNormLayer &&) = delete;

        /** Default destructor */
        ~BINERMSNormLayer();

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
        void configure(const BIITensor *input, const BIITensor *gamma, BIITensor *output);

        void dynamic_configure(const BIITensor *input) const;

        static BIStatus
        validate(const BIITensorInfo *input, const BIITensorInfo *gamma, const BIITensorInfo *output);

        void run() override;

    private:
        BIMemoryGroup _memory_group; /**< Function memory group */
        std::unique_ptr<cpu::BINERMSNormLayerKernel> _rms_norm_kernel; /**<RMS Normalization layer kernel */
        BITensor _input;
        BITensor _gamma;
    };
}
