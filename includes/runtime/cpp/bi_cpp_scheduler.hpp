//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_CPP_SCHEDULER_HPP
#define BATMANINFER_BI_CPP_SCHEDULER_HPP

#include <data/core/experimental/types.hpp>
#include <runtime/bi_i_scheduler.hpp>

#include <memory>

namespace BatmanInfer {
    /** C++11 实现的线程池，用于自动将内核的执行分配到多个线程中。
     *
     * 它有两种调度模式：线性（Linear）和扇出（Fanout）（详细信息请参考实现）。
     * 调度模式会根据运行时环境自动选择。然而，也可以通过环境变量 ARM_COMPUTE_CPP_SCHEDULER_MODE 强制指定模式，例如：
     * BI_COMPUTE_CPP_SCHEDULER_MODE=linear      # 强制选择线性调度模式
     * BI_COMPUTE_CPP_SCHEDULER_MODE=fanout      # 强制选择扇出调度模式
     */
    class BICPPScheduler final : public BIIScheduler {
    public:
        BICPPScheduler();

        ~BICPPScheduler();

        /**
         * @brief 获取调度器的单例模式
         *
         * @note this method has been deprecated and will be remover in future releases
         *
         * @return
         */
        static BICPPScheduler &get();

        void set_num_threads(unsigned int num_threads) override;

        void set_num_threads_with_affinity(unsigned int num_threads, BatmanInfer::BIIScheduler::BindFunc func) override;

        unsigned int num_threads() const override;

        void schedule(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints) override;

        void schedule_op(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints,
                         const BatmanInfer::BIWindow &window, BatmanInfer::BIITensorPack &tensors) override;

        void schedule_kv_split(BIITensorPack &tensors) override;

        void schedule_kv_concat(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst) override;

    protected:
        /**
         * @brief 将使用 num_threads 并行运行工作负载。
         * @param workloads
         */
        void run_workloads(std::vector<BIWorkload> &workloads) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}


#endif //BATMANINFER_BI_CPP_SCHEDULER_HPP
