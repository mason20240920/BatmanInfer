//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_SINGLE_THREAD_SCHEDULER_HPP
#define BATMANINFER_BI_SINGLE_THREAD_SCHEDULER_HPP

#include <runtime/bi_i_scheduler.hpp>

namespace BatmanInfer {
    class BISingleThreadScheduler final : public BIIScheduler {
    public:
        BISingleThreadScheduler() = default;

        /**
         * @brief 设置调度器用于运行内核的线程数量。
         * @param num_threads 对于该调度器，此参数会被忽略，因为线程数量始终为 1。
         */
        void set_num_threads(unsigned int num_threads) override;

        /**
         * @brief 返回 SingleThreadScheduler 中的线程数量，该值始终为 1
         * @return
         */
        unsigned int num_threads() const override;

        /**
         * @brief 在调用者的线程中同步运行内核
         * @param kernel
         * @param hints
         */
        void schedule(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints) override;

        /**
         * @brief 在调用者的线程中同步运行内核
         * @param kernel
         * @param hints
         * @param window
         * @param tensors
         */
        void schedule_op(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints,
                         const BatmanInfer::BIWindow &window, BatmanInfer::BIITensorPack &tensors) override;

        void schedule_kv(BIITensorPack &tensors) override;

    protected:
        /**
         * @brief 将按顺序依次运行工作负载
         * @param workloads
         */
        void run_workloads(std::vector<BIWorkload> &workloads) override;
    };
}

#endif //BATMANINFER_BI_SINGLE_THREAD_SCHEDULER_HPP
