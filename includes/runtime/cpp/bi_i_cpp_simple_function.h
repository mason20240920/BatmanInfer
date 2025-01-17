//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/cpp/bi_i_cpp_kernel.hpp"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

    /** Basic interface for functions which have a single CPP kernel */
    class BIICPPSimpleFunction : public BIIFunction
    {
    public:
        /** Constructor */
        BIICPPSimpleFunction();

        // Inherited methods overridden:
        void run() override final;

    protected:
        std::unique_ptr<BIICPPKernel> _kernel; /**< Kernel to run */
    };

} // namespace BatmanInfer
