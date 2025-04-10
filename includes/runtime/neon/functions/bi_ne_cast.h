//
// Created by holynova on 25-4-10.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"
#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    class BINECast : public BIIFunction {
    public:
        BINECast();
        ~BINECast() override;

        BINECast(const BINECast &) = delete;
        BINECast(BINECast &&) noexcept;

        BINECast &operator=(const BINECast &) = delete;
        BINECast &operator=(BINECast &&) noexcept;

        void configure(BIITensor *input, BIITensor *output, BIConvertPolicy policy);

        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output, BIConvertPolicy policy);

        // Inherited methods overridden
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
