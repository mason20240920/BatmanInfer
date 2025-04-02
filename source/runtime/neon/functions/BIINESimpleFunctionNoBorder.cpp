//
// Created by Mason on 2025/4/1.
//

#include <runtime/neon/bi_i_ne_simple_function_no_border.h>

#include <data/core/bi_window.hpp>
#include <runtime/utils.hpp>
#include <data/core/cpp/bi_i_cpp_kernel.hpp>


namespace BatmanInfer {
    BIINESimpleFunctionNoBorder::~BIINESimpleFunctionNoBorder() = default;

    BIINESimpleFunctionNoBorder::BIINESimpleFunctionNoBorder(BIIRuntimeContext *ctx) : _kernel(), _ctx(ctx) {
    }

    void BIINESimpleFunctionNoBorder::run() {
        utils::schedule_kernel_on_ctx(_ctx, _kernel.get(), BIWindow::DimY);
    }
}
