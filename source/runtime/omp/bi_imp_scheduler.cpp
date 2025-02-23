//
// Created by Mason on 2025/1/4.
//

#include <runtime/omp/bi_imp_scheduler.hpp>

#include <data/core/cpp/bi_i_cpp_kernel.hpp>
#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_utils.hpp>

#include <omp.h>

namespace BatmanInfer {

#if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)
    BIOMPScheduler::BIOMPScheduler() :  _num_threads(cpu_info().get_cpu_num_excluding_little()),
                                        _nonlittle_num_cpus(cpu_info().get_cpu_num_excluding_little()) {}
#else

    BIOMPScheduler::BIOMPScheduler() : _num_threads(omp_get_max_threads()),
                                       _nonlittle_num_cpus(cpu_info().get_cpu_num_excluding_little()) {}

#endif

    unsigned int BIOMPScheduler::num_threads() const {
        return _num_threads;
    }

    void BIOMPScheduler::set_num_threads(unsigned int num_threads) {
        const unsigned int num_cores = omp_get_max_threads();
#if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)
        const unsigned int adjusted_num_threads = std::min(_nonlittle_num_cpus, num_threads);
    _num_threads                            = (num_threads == 0) ? num_cores : adjusted_num_threads;
#else  /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
        _num_threads = (num_threads == 0) ? num_cores : num_threads;
#endif /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
    }

    void BIOMPScheduler::schedule(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints) {
        BIITensorPack tensors;
        schedule_common(kernel, hints, kernel->window(), tensors);
    }

    void BIOMPScheduler::schedule_op(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints,
                                     const BatmanInfer::BIWindow &window, BatmanInfer::BIITensorPack &tensors) {
        // The rest of the logic in this function does not handle the
        // split_dimensions_all case so we defer to IScheduler::schedule_common()
        if (hints.split_dimension() == BIIScheduler::split_dimensions_all) {
            return schedule_common(kernel, hints, window, tensors);
        }

        BI_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
        BI_COMPUTE_ERROR_ON_MSG(hints.strategy() == BIStrategyHint::DYNAMIC,
                                "Dynamic scheduling is not supported in OMPScheduler");

        const BIWindow &max_window = window;
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        const unsigned int mws = kernel->get_mws(CPUInfo::get(), _num_threads);

        // Ensure each thread has mws amount of work to do (i.e. ceil(num_iterations / mws) threads)
        const unsigned int candidate_num_threads = (num_iterations + mws - 1) / mws;

        // Cap the number of threads to be spawn with the size of the thread pool
        const unsigned int num_threads = std::min(candidate_num_threads, _num_threads);

        if (!kernel->is_parallelisable() || num_threads == 1) {
            ThreadInfo info;
            info.cpu_info = &cpu_info();
            kernel->run_op(tensors, max_window, info);
        } else {
            const unsigned int num_windows = num_threads;
            std::vector<BIIScheduler::BIWorkload> workloads(num_windows);
            for (unsigned int t = 0; t < num_windows; t++) {
                //Capture 't' by copy, all the other variables by reference:
                workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo &info) {
                    BIWindow win = max_window.split_window(hints.split_dimension(), t, num_windows);
                    win.validate();
                    kernel->run_op(tensors, win, info);
                };
            }
            run_workloads(workloads);
        }
    }

#ifndef DOXYGEN_SKIP_THIS

    void BIOMPScheduler::run_workloads(std::vector<BIWorkload> &workloads) {
        // 计算任务的总数，即 workloads 的大小，并将其存储在 amount_of_work 中
        const auto amount_of_work = static_cast<unsigned int>(workloads.size());
        // 线程数不能超过任务数，因此取两者的最小值
        const unsigned int num_threads_to_use = std::min(_num_threads, amount_of_work);

        if (num_threads_to_use < 1)
            return;

        // 存储线程相关的信息
        ThreadInfo info;
        // 当前 CPU 信息的指针
        info.cpu_info = &cpu_info();
        // 实际使用的线程数
        info.num_threads = static_cast<int>(num_threads_to_use);

        // 在非 Android 环境中，omp_num_threads 被设置为 _num_threads，即系统支持的最大线程数
#if !defined(__ANDROID__)
        // Use fixed number of omp threads in the thread pool because changing this
        // in-between kernel execution negatively affects the scheduler performance,
        // possibly switching between X and Y number of threads, causing reconfiguration
        // of the synchronization mechanism. This has been only tested in a subset of
        // operating systems, thus we limit the change using guards.
        const unsigned int omp_num_threads = _num_threads;
#else  /* !__ANDROID__ */
        const unsigned int omp_num_threads = num_threads_to_use;
#endif /* __ANDROID__ */

        /**
         * @brief omp parallel for：并行化 for 循环。
         * firstprivate(info)：每个线程都会复制一份 info 对象（每个线程有独立的副本）。
         * num_threads(omp_num_threads)：指定 OpenMP 使用的线程数。
         * default(shared)：默认情况下，所有变量在线程之间共享（除非显式指定为私有）。
         * proc_bind(close)：绑定线程到接近的处理器核，以减少线程迁移的开销。
         * schedule(static, 1)：静态调度，每个线程分配固定数量的迭代（这里每次分配 1 个任务）
         */
#pragma omp parallel for firstprivate(info) num_threads(omp_num_threads) default(shared) proc_bind(close) \
    schedule(static, 1)
        // 遍历所有任务的索引 wid，范围是 [0, amount_of_work)
        for (unsigned int wid = 0; wid < amount_of_work; ++wid) {
            // 获取当前线程的 ID（tid），这是 OpenMP 提供的函数
            const int tid = omp_get_thread_num();

            // 当前线程的 ID 存储到 info.thread_id 中
            info.thread_id = tid;
            // 执行工作负载
            workloads[wid](info);
        }
    }

#endif
}