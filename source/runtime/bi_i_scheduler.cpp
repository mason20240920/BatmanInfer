//
// Created by Mason on 2025/1/4.
//

#include <runtime/bi_i_scheduler.hpp>

#include <common/cpu_info/cpu_info.hpp>
#include <data/core/bi_window.hpp>
#include <runtime/bi_scheduler_utils.hpp>
#include <data/core/cpp/bi_i_cpp_kernel.hpp>

#include <data/core/utils/logging/bi_logging_log.hpp>

namespace BatmanInfer {
    BIIScheduler::BIIScheduler() {
        // 计算出最佳的执行线程数量
        _num_threads_hint = cpu_info::num_threads_hint();
    }

    CPUInfo &BIIScheduler::cpu_info() {
        return CPUInfo::get();
    }

    void
    BIIScheduler::set_num_threads_with_affinity(unsigned int num_threads, BatmanInfer::BIIScheduler::BindFunc func) {
        BI_COMPUTE_UNUSED(num_threads, func);
        BI_COMPUTE_ERROR("Feature for affinity setting is not implemented");
    }

    unsigned int BIIScheduler::num_threads_hint() const {
        return _num_threads_hint;
    }

    void BIIScheduler::schedule_common(BIICPPKernel *kernel,
                                       const BIIScheduler::Hints &hints,
                                       const BIWindow &window,
                                       BIITensorPack &tensors) {
        BI_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
#ifndef BARE_METAL
        const BIWindow &max_window = window;
        // 当 split_dimensions_all 被设置时的处理
        if (hints.split_dimension() == BIIScheduler::split_dimensions_all) {
            /**
             * 如果分割维度是 size_t 的最大值，则这表示我们应该并行化
             * 所有维度
             */
            const std::size_t m = max_window.num_iterations(BIWindow::DimX);
            const std::size_t n = max_window.num_iterations(BIWindow::DimY);
            //in c++17 this can be swapped for   auto [ m_threads, n_threads ] = split_2d(...
            unsigned m_threads, n_threads;
            std::tie(m_threads, n_threads) = scheduler_utils::split_2d(this->num_threads(), m, n);

            std::vector<BIIScheduler::BIWorkload> workloads;
            for (unsigned int ni = 0; ni != n_threads; ++ni) {
                for (unsigned int mi = 0; mi != m_threads; ++mi) {
                    workloads.push_back(
                        [ni, mi, m_threads, n_threads, &max_window, &kernel](const ThreadInfo &info) {
                            //narrow the window to our mi-ni workload
                            BIWindow win = max_window.split_window(BIWindow::DimX, mi, m_threads)
                                    .split_window(BIWindow::DimY, ni, n_threads);

                            win.validate();

                            BIWindow thread_locator;
                            thread_locator.set(BIWindow::DimX, BIWindow::BIDimension(mi, m_threads));
                            thread_locator.set(BIWindow::DimY, BIWindow::BIDimension(ni, n_threads));

                            thread_locator.validate();

                            kernel->run_nd(win, info, thread_locator);
                        });
                }
            }
            run_workloads(workloads);
        } else {
            // 如果 hints.split_dimension() 指定了某个维度（如 DimX 或 DimY），则仅在该维度上并行化
            const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
            const unsigned int num_threads = std::min(num_iterations, this->num_threads());

            if (num_iterations == 0)
                return;

            if (!kernel->is_parallelisable() || num_threads == 1) {
                ThreadInfo info;
                info.cpu_info = &cpu_info();
                if (tensors.empty())
                    kernel->run(max_window, info);
                else
                    kernel->run_op(tensors, max_window, info);
            } else {
                unsigned int num_windows = 0;
                switch (hints.strategy()) {
                    case BIStrategyHint::STATIC:
                        num_windows = num_threads;
                        break;
                    case BIStrategyHint::DYNAMIC: {
                        const unsigned int granule_threshold =
                                (hints.threshold() <= 0)
                                    ? num_threads
                                    : static_cast<unsigned int>(hints.threshold());
                        // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                        num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
                        break;
                    }
                    default:
                        BI_COMPUTE_ERROR("Unknown strategy");
                }

                // 保证最小的窗口大于最小的工作负载大小
                num_windows = adjust_num_of_windows(max_window, hints.split_dimension(), num_windows, *kernel,
                                                    cpu_info());

                std::vector<BIIScheduler::BIWorkload> workloads(num_windows);
                for (unsigned int t = 0; t < num_windows; ++t) {
                    //Capture 't' by copy, all the other variables by reference:
                    workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo &info) {
                        BIWindow win = max_window.split_window(hints.split_dimension(), t, num_windows);
                        win.validate();

                        if (tensors.empty()) {
                            kernel->run(win, info);
                        } else {
                            kernel->run_op(tensors, win, info);
                        }
                    };
                }
                run_workloads(workloads);
            }
        }
#else
        BI_COMPUTE_UNUSED(kernel, hints, window, tensors);
#endif
    }

    void BIIScheduler::run_tagged_workloads(std::vector<BIWorkload> &workloads, const char *tag) {
        BI_COMPUTE_UNUSED(tag);
        run_workloads(workloads);
    }

    std::size_t BIIScheduler::adjust_num_of_windows(const BatmanInfer::BIWindow &window, std::size_t split_dimension,
                                                    std::size_t init_num_windows,
                                                    const BatmanInfer::BIICPPKernel &kernel,
                                                    const BatmanInfer::CPUInfo &cpu_info) {
        // 缓解狭窄分割问题，该问题发生在分割维度过小而无法分割时（因此称为“狭窄”）。
        if (window.num_iterations(split_dimension) < init_num_windows) {
            auto recommended_split_dim = BIWindow::DimX;
            for (std::size_t dims = BIWindow::DimY; dims <= BIWindow::DimW; ++dims) {
                if (window.num_iterations(recommended_split_dim) < window.num_iterations(dims))
                    recommended_split_dim = dims;
            }
            BI_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(
                "%zu dimension is not a suitable dimension to split the workload. Recommended: %zu recommended_split_dim",
                split_dimension, recommended_split_dim);
        }

        for (auto t = init_num_windows; t > 0; --t) // Trying the highest number of windows ,init_num_windows, first
        {
            // Try splitting the workload into t, subject to each subworkload size <= mws.
            if ((window.num_iterations(split_dimension) / kernel.get_mws(cpu_info, t)) >= t) {
                if (t != init_num_windows) {
                    BI_COMPUTE_LOG_INFO_MSG_CORE(
                        "The scheduler is using a different thread count than the one assigned by the user.");
                }
                return t;
            }
        }
        BI_COMPUTE_LOG_INFO_MSG_CORE(
            "The scheduler is using single thread instead of the thread count assigned by the user.");
        return 1; //  If the workload is so small that it can't be split, we should run a single thread
    }
}
