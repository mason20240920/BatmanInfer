//
// Created by Mason on 2025/1/19.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>

#include <memory>

namespace BatmanInfer {
    class BIITensor;

    class BIITensorInfo;

    /**
     * Basic function to run cpu::kernels::BICpuCopyKernel
     */
    class BINECopy : public BIIFunction {
    public:
        BINECopy();

        ~BINECopy();

        BINECopy(const BINECopy &) = delete;

        BINECopy(BINECopy &&);

        BINECopy &operator=(const BINECopy &) = delete;

        BINECopy &operator=(BINECopy &&);

        /**
         * Initialise the function's source and destination.
         *
         * Valid data layouts:
         * - All
         *
         * @param input
         * @param output
         */
        void configure(BIITensor *input,
                       BIITensor *output);

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *output);

        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}