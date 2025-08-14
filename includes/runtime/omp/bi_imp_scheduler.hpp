//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_IMP_SCHEDULER_HPP
#define BATMANINFER_BI_IMP_SCHEDULER_HPP

#include <runtime/bi_i_scheduler.hpp>

namespace BatmanInfer {
    /**
     * @brief 线程池用于自动将内核的执行分配到多个线程中。
     */
    class BIOMPScheduler final : public BIIScheduler {
    public:
        BIOMPScheduler();

        /**
         * @brief 设置调度器用于运行内核的线程数量
         * @param num_threads 如果设置为 0，则使用 `omp_get_max_threads()` 返回的线程数量；
         *        否则，使用指定的线程数量。
         */
        void set_num_threads(unsigned int num_threads) override;

        /**
         * @brief 返回 OMPScheduler 池中可用的线程数量。
         * @return OMPScheduler 中可用的线程数量。
         */
        unsigned int num_threads() const override;

        /**
         * @brief  如果可能，多线程执行传入的内核
         *
         * 如果以下任一条件为真，内核将仅在单线程上运行：
         * - `ICPPKernel::is_parallelisable()` 返回 `false`
         * - 调度器已初始化为仅使用一个线程。
         * @param kernel 要执行的内核
         * @param hints 调度器的提示信息。
         */
        void schedule(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints) override;

        /**
         * @brief 如果可能，多线程执行传入的内核
         *
         *  * 如果以下任一条件为真，内核将仅在单线程上运行：
         * - `ICPPKernel::is_parallelisable()` 返回 `false`
         * - 调度器已初始化为仅使用一个线程。
         * @param kernel 要执行的内核
         * @param hints 调度器的提示信息
         * @param window 用于内核执行的窗口
         * @param tensors 要操作的张量向量
         */
        void schedule_op(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints,
                         const BatmanInfer::BIWindow &window, BatmanInfer::BIITensorPack &tensors) override;

        void schedule_kv_split(BIITensorPack &tensors,const std::vector<size_t>& ava_len) override;


        void schedule_kv_concat(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst, const std::vector<size_t> &ava_len, int layer_idx) override;;

        void schedule_kv_full_fill(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst, const std::vector<size_t> &ava_len) override;

        void schedule_change_q(BIITensorPack &tensors, const std::vector<size_t> &ava_len, size_t max_seq_len) override;

    protected:
        /**
         * @brief 执行所有传入的工作负载
         *
         * @note 无法保证这些工作负载的执行顺序，也无法保证它们是否会并行执行。
         *
         * @param workloads 要运行的工作负载数组。
         */
        void run_workloads(std::vector<BIWorkload> &workloads) override;

    private:
        unsigned int _num_threads;
        unsigned int _nonlittle_num_cpus;
    };
}

#endif //BATMANINFER_BI_IMP_SCHEDULER_HPP
