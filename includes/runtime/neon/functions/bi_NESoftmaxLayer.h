//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_error.h"
#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;

    /** Basic function to compute a SoftmaxLayer and a Log SoftmaxLayer. */
    template<bool IS_LOG = false>
    class BINESoftmaxLayerGeneric : public BIIFunction {
    public:
        /** Constructor */
        BINESoftmaxLayerGeneric(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINESoftmaxLayerGeneric() : BINESoftmaxLayerGeneric(BIMemoryManagerOnDemand::make_default()) {
        }

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINESoftmaxLayerGeneric(const BINESoftmaxLayerGeneric &) = delete;

        /** Default move constructor */
        BINESoftmaxLayerGeneric(BINESoftmaxLayerGeneric &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINESoftmaxLayerGeneric &operator=(const BINESoftmaxLayerGeneric &) = delete;

        /** Default move assignment operator */
        BINESoftmaxLayerGeneric &operator=(BINESoftmaxLayerGeneric &&);

        /** Default destructor */
        ~BINESoftmaxLayerGeneric();

        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |QASYMM8        |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |
         * |F16            |F16            |
         * |F32            |F32            |
         *
         * @param[in,out] input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32. If the width is not a
         *                       multiple of the internal processing block size, @ref NEFillBorder replicates the
         *                       last value of each row to the nearest multiple.
         * @param[out]    output Destination tensor. Data types supported: same as @p input.
         * @param[in]     beta   (Optional) A scaling factor for the exponent.
         * @param[in]     axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
         *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
         */
        void configure(BIITensor *input, BIITensor *output, float beta = 1.0f, int32_t axis = 0);

        /**
         * 用于在动态改变 input 与 output 的 shape 后调整算子内部信息
         */
        void dynamic_configure();

        /** Static function to check if given info will lead to a valid configuration of @ref BINESoftmaxLayer
         *
         * @param[in] input  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
         * @param[in] output Destination tensor info. Data types supported: same as @p input
         * @param[in] beta   (Optional) A scaling factor for the exponent.
         * @param[in] axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
         *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
         *
         * @return a status
         */
        static BIStatus
        validate(const BIITensorInfo *input, const BIITensorInfo *output, float beta = 1.0f, int32_t axis = 0);

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

    using BINESoftmaxLayer = BINESoftmaxLayerGeneric<false>;
    using BINELogSoftmaxLayer = BINESoftmaxLayerGeneric<true>;

} // namespace BatmanInfer
