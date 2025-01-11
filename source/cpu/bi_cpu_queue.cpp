//
// Created by Mason on 2025/1/11.
//

#include <cpu/bi_cpu_queue.hpp>

#include <runtime/bi_scheduler.hpp>

namespace BatmanInfer {
    namespace cpu {
        BICpuQueue::BICpuQueue(BatmanInfer::BIIContext *ctx, const BclQueueOptions *options) : BIIQueue(ctx) {
            BI_COMPUTE_UNUSED(options);
        }

        BatmanInfer::BIIScheduler &BICpuQueue::scheduler() {
            return BatmanInfer::BIScheduler::get();
        }

        StatusCode BICpuQueue::finish() {
            return StatusCode::Success;
        }
    }
}