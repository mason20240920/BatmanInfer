//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_runtime_context.hpp"

#include <memory>

namespace BatmanInfer {
    class BIICPPKernel;
    using INEKernel = BIICPPKernel;

    /** Basic interface for functions which have a single CPU kernel and no border */
    class BIINESimpleFunctionNoBorder : public BIIFunction {
    public:
        /** Constructor
         *
         * @param[in] ctx Runtime context to be used by the function
         */
        BIINESimpleFunctionNoBorder(BIIRuntimeContext *ctx = nullptr);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIINESimpleFunctionNoBorder(const BIINESimpleFunctionNoBorder &) = delete;

        /** Default move constructor */
        BIINESimpleFunctionNoBorder(BIINESimpleFunctionNoBorder &&) = default;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIINESimpleFunctionNoBorder &operator=(const BIINESimpleFunctionNoBorder &) = delete;

        /** Default move assignment operator */
        BIINESimpleFunctionNoBorder &operator=(BIINESimpleFunctionNoBorder &&) = default;

        /** Default destructor */
        ~BIINESimpleFunctionNoBorder();

        // Inherited methods overridden:
        void run();

    protected:
        std::unique_ptr<INEKernel> _kernel; /**< Kernel to run */
        BIIRuntimeContext *_ctx; /**< Context to use */
    };
} // namespace BatmanInfer
