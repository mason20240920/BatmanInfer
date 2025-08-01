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

    /** Basic function to run cpu::kernels::CpuPermuteKernel */
    class BINEPermute : public BIIFunction {
    public:
        /** Default Constructor */
        BINEPermute();

        /** Default Destructor */
        ~BINEPermute();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEPermute(const BINEPermute &) = delete;

        /** Default move constructor */
        BINEPermute(BINEPermute &&) = default;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEPermute &operator=(const BINEPermute &) = delete;

        /** Default move assignment operator */
        BINEPermute &operator=(BINEPermute &&) = default;

        /** Configure the permute function
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src    |dst    |
         * |:------|:------|
         * |All    |All    |
         *
         * @note Arbitrary permutation vectors are supported with rank not greater than 4
         *
         * @param[in]  input  The input tensor to permute. Data types supported: All
         * @param[out] output The output tensor. Data types supported: Same as @p input
         * @param[in]  perm   Permutation vector
         */
        void configure(const BIITensor *input, BIITensor *output, const PermutationVector &perm);

        void dynamic_configure(const BIITensor *input,
                               BIITensor *output) const;

        /** Static function to check if given info will lead to a valid configuration of @ref NEPermute
         *
         * @note Arbitrary permutation vectors are supported with rank not greater than 4
         *
         * @param[in] input  The input tensor to permute. Data types supported: All
         * @param[in] output The output tensor. Data types supported: Same as @p input
         * @param[in] perm   Permutation vector
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output,
                                 const PermutationVector &perm);

        // Inherited methods overridden
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace BatmanInfer
