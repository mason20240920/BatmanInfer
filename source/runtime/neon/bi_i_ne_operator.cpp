//
// Created by Mason on 2025/1/3.
//

#include <runtime/neon/bi_i_ne_operator.hpp>

#include <data/core/bi_window.hpp>
#include <data/core/neon/bi_i_ne_kernel.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

namespace BatmanInfer {
    namespace experimental {
        BIINEOperator::~BIINEOperator() = default;

        BIINEOperator::BIINEOperator(BIIRuntimeContext *ctx) : _kernel(), _ctx(ctx), _workspace() {

        }

        void BIINEOperator::run(BIITensorPack &tensors) {
            if (tensors.empty())
                BI_COMPUTE_ERROR("No inputs provided");

            run(tensors, _kernel->window());
        }

        void BIINEOperator::run(BIITensorPack &tensors, const BIWindow &window) {
            BINEScheduler::get().schedule_op(_kernel.get(), BIWindow::DimY, window, tensors);
        }

        void BIINEOperator::prepare(BatmanInfer::BIITensorPack &constants) {
            BI_COMPUTE_UNUSED(constants);
        }

        BIMemoryRequirements BIINEOperator::workspace() const {
            return _workspace;
        }
    }
}