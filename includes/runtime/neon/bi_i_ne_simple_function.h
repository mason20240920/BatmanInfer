//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

    class BIICPPKernel;
    class BINEFillBorderKernel;
    using INEKernel = BIICPPKernel;

    /** Basic interface for functions which have a single CPU kernel */
    class BIINESimpleFunction : public BIIFunction
    {
    public:
        /** Constructor */
        BIINESimpleFunction();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIINESimpleFunction(const BIINESimpleFunction &) = delete;
        /** Default move constructor */
        BIINESimpleFunction(BIINESimpleFunction &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIINESimpleFunction &operator=(const BIINESimpleFunction &) = delete;
        /** Default move assignment operator */
        BIINESimpleFunction &operator=(BIINESimpleFunction &&) = default;
        /** Default destructor */
        ~BIINESimpleFunction();

        // Inherited methods overridden:
        void run() override final;

    protected:
        std::unique_ptr<INEKernel>            _kernel;         /**< Kernel to run */
        std::unique_ptr<BINEFillBorderKernel> _border_handler; /**< Kernel to handle image borders */
    };

} // namespace BatmanInfer
