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
    class BIStatus;

    /** Basic function to execute concatenate tensors along a given axis */
    class BINEConcatenateLayer : public BIIFunction
    {
    public:
        /** Default constructor */
        BINEConcatenateLayer();
        /** Destructor */
        ~BINEConcatenateLayer();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEConcatenateLayer(const BINEConcatenateLayer &) = delete;
        /** Default move constructor */
        BINEConcatenateLayer(BINEConcatenateLayer &&);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEConcatenateLayer &operator=(const BINEConcatenateLayer &) = delete;
        /** Default move assignment operator */
        BINEConcatenateLayer &operator=(BINEConcatenateLayer &&);
        /** Initialise the kernel's inputs vector and output.
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
         * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
         * @note Preconditions can be found respectively at cpu::kernels::CpuConcatenateWidthKernel, cpu::kernels::CpuConcatenateHeightKernel,
         *       cpu::kernels::CpuConcatenateDepthKernel and cpu::kernels::CpuConcatenateBatchKernel.
         *
         * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
         * @param[out]    output        Output tensor. Data types supported: Same as @p input.
         * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
         */
        void configure(std::vector<const BIITensor *> inputs_vector, BIITensor *output, size_t axis);
        /** Static function to check if given info will lead to a valid configuration of @ref NEConcatenateLayer
         *
         * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
         * @note Preconditions can be found respectively at cpu::kernels::CpuConcatenateWidthKernel, cpu::kernels::CpuConcatenateHeightKernel,
         *       cpu::kernels::CpuConcatenateDepthKernel and cpu::kernels::CpuConcatenateBatchKernel.
         *
         * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
         * @param[in] output        Output tensor info. Data types supported: Same as @p input.
         * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
         *
         * @return a status
         */
        static BIStatus
        validate(const std::vector<const BIITensorInfo *> &inputs_vector, const BIITensorInfo *output, size_t axis);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
