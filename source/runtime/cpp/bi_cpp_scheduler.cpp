//
// Created by Mason on 2025/1/4.
//

#include <runtime/cpp/bi_cpp_scheduler.hpp>

#include <data/core/bi_error.h>
#include <data/core/utils/misc/utils.hpp>
#include <data/core/utils/logging/bi_logging_log.hpp>

#include <data/core/cpp/bi_i_cpp_kernel.hpp>

#include <atomic>
#include <thread>
#include <list>

namespace BatmanInfer {
    namespace {
        class BIThreadFeeder {
        public:
            /**
             * @brief
             * @param start 起始任务索引。
             * @param end 结束条件（`get_next()` 返回的最后一个值为 `end - 1`）。
             */
            explicit BIThreadFeeder(unsigned int start = 0,
                                    unsigned int end = 0) : _atomic_counter(start),
                                                            _end(end) {
            }

            /**
             * @brief 获取下一个任务索引。
             * @param next 如果存在下一个任务索引，将其存储在 `next` 中。
             * @return 如果任务范围未结束，返回 `true`；否则返回 `false`。
             */
            bool get_next(unsigned int &next) {
                next = atomic_fetch_add_explicit(&_atomic_counter, 1u, std::memory_order_relaxed);
                return next < _end;
            }

        private:
            // 当前任务索引，线程安全
            std::atomic_uint _atomic_counter;
            // 任务范围的结束值
            const unsigned int _end;
        };


        /**
         * @brief 先执行 `workloads[info.thread_id]`，然后调用 `feeder` 获取下一个要执行的工作负载索引。
         *
         * 将持续运行工作负载，直到 `feeder` 达到其范围的结束。
         *
         * @param workloads 工作负载的数组。
         * @param feeder 指示下一个要执行的工作负载的 `feeder`。
         * @param info 线程和 CPU 的相关信息。
         */
        void process_workloads(std::vector<BIIScheduler::BIWorkload> &workloads,
                               BIThreadFeeder &feeder,
                               const ThreadInfo &info) {
            unsigned int workload_index = info.thread_id;
            do {
                BI_COMPUTE_ERROR_ON(workload_index >= workloads.size());
                workloads[workload_index](info);
            } while (feeder.get_next(workload_index));
        }

        /**
         * @brief  设置线程亲和性。将当前线程绑定到特定的核心上
         * @param core_id
         */
        void set_thread_affinity(int core_id) {
            if (core_id < 0) {
                return;
            }

#if !defined(_WIN64) && !defined(__APPLE__) && !defined(__OpenBSD__)
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core_id, &set);
            BI_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
#endif /* !defined(__APPLE__) && !defined(__OpenBSD__) */
        }

        /*
 * CPPScheduler 当前支持两种调度模式：
 *
 * Linear（线性模式）：
 *  这是默认模式，在该模式下，所有的调度任务由主线程以线性方式（通过循环）完成。
 *  例如：如果总共有 8 个线程，则线程池中有 7 个线程，主线程负责启动线程池中的所有其他线程。
 *  这种模式简单高效，适合线程池较小或任务规模较小且均匀的场景。
 *
 * Fanout（分散模式）：
 *  在分散模式下，调度任务（启动其他线程）由多个线程分担，而不仅仅依赖主线程。
 *  这种模式通过多线程分担调度任务，减轻主线程的负担，能够更好地利用多核 CPU 的性能，适合线程池较大或任务动态分布的场景。
 *
 *  调度器有一个固定参数：wake_fanout，调度过程如下：
 *  1. 主线程唤醒线程池中的前 wake_fanout - 1 个 FanoutThread：
 *      从线程：0
 *      到线程（不包含）：wake_fanout - 1
 *  2. 每个 FanoutThread 接着唤醒线程池中 wake_fanout 个 FanoutThread：
 *      从线程：(i + 1) * wake_fanout - 1
 *      到线程（不包含）：(i + 2) * wake_fanout - 1
 *      其中，i 是当前线程的线程 ID。
 *      结束位置会被限制在线程池大小 / 使用线程数 - 1。
 *
 *  例如：对于总线程数为 8（1 个主线程和线程池中的 7 个 FanoutThread），并且 fanout 参数为 3：
 *  1. 主线程唤醒 FanoutThread 0 和 1。
 *  2. FanoutThread 0 唤醒 FanoutThread 2、3、4。
 *  3. FanoutThread 1 唤醒 FanoutThread 5、6。
 *
 *  Linear 模式适合小规模、均匀的计算任务，而 Fanout 模式则更适合大规模、动态分布的任务。
 */
        class BIThread final {
        public:
            /** Start a new thread
             *
             * Thread will be pinned to a given core id if value is non-negative
             *
             * @param[in] core_pin Core id to pin the thread on. If negative no thread pinning will take place
             */
            explicit BIThread(int core_pin = -1);

            BIThread(const BIThread &) = delete;

            BIThread &operator=(const BIThread &) = delete;

            BIThread(BIThread &&) = delete;

            BIThread &operator=(BIThread &&) = delete;

            /** Destructor. Make the thread join. */
            ~BIThread();

            /** Set workloads */
            void set_workload(std::vector<BIIScheduler::BIWorkload> *workloads, BIThreadFeeder &feeder,
                              const ThreadInfo &info);

            /** Request the worker thread to start executing workloads.
             *
             * The thread will start by executing workloads[info.thread_id] and will then call the feeder to
             * get the index of the following workload to run.
             *
             * @note This function will return as soon as the workloads have been sent to the worker thread.
             * wait() needs to be called to ensure the execution is complete.
             */
            void start();

            /** Wait for the current kernel execution to complete. */
            std::exception_ptr wait();

            /** Function ran by the worker thread. */
            void worker_thread();

            /** Set the scheduling strategy to be linear */
            void set_linear_mode() {
                _thread_pool = nullptr;
                _wake_beg = 0;
                _wake_end = 0;
            }

            /** Set the scheduling strategy to be fanout */
            void set_fanout_mode(std::list<BIThread> *thread_pool, unsigned int wake_beg, unsigned int wake_end) {
                _thread_pool = thread_pool;
                _wake_beg = wake_beg;
                _wake_end = wake_end;
            }

        private:
            std::thread _thread{};
            ThreadInfo _info{};
            std::vector<BIIScheduler::BIWorkload> *_workloads{nullptr};
            BIThreadFeeder *_feeder{nullptr};
            std::mutex _m{};
            std::condition_variable _cv{};
            bool _wait_for_work{false};
            bool _job_complete{true};
            std::exception_ptr _current_exception{nullptr};
            int _core_pin{-1};
            std::list<BIThread> *_thread_pool{nullptr};
            unsigned int _wake_beg{0};
            unsigned int _wake_end{0};
        };

        BIThread::BIThread(int core_pin) : _core_pin(core_pin) {
            _thread = std::thread(&BIThread::worker_thread, this);
        }

        BIThread::~BIThread() {
            // Make sure worker thread has ended
            if (_thread.joinable()) {
                BIThreadFeeder feeder;
                set_workload(nullptr, feeder, ThreadInfo());
                start();
                _thread.join();
            }
        }

        void
        BIThread::set_workload(std::vector<BIIScheduler::BIWorkload> *workloads, BatmanInfer::BIThreadFeeder &feeder,
                               const BatmanInfer::ThreadInfo &info) {
            _workloads = workloads;
            _feeder = &feeder;
            _info = info;
        }

        void BIThread::start() { {
                std::lock_guard<std::mutex> lock(_m);
                _wait_for_work = true;
                _job_complete = false;
            }
            _cv.notify_one();
        }

        std::exception_ptr BIThread::wait() { {
                std::unique_lock<std::mutex> lock(_m);
                _cv.wait(lock, [&] { return _job_complete; });
            }
            return _current_exception;
        }

        void BIThread::worker_thread() {
            set_thread_affinity(_core_pin);

            while (true) {
                std::unique_lock<std::mutex> lock(_m);
                _cv.wait(lock, [&] { return _wait_for_work; });
                _wait_for_work = false;

                _current_exception = nullptr;

                // Exit if the worker thread has not been fed with workloads
                if (_workloads == nullptr || _feeder == nullptr) {
                    return;
                }

                // Wake up more peer threads from thread pool if this job has been delegated to the current thread
                if (_thread_pool != nullptr) {
                    auto thread_it = _thread_pool->begin();
                    std::advance(thread_it, std::min(static_cast<unsigned int>(_thread_pool->size()), _wake_beg));
                    auto wake_end = std::min(_wake_end, static_cast<unsigned int>(_info.num_threads - 1));
                    for (unsigned int t = _wake_beg; t < wake_end; ++t, ++thread_it) {
                        thread_it->start();
                    }
                }
#ifndef BI_COMPUTE_EXCEPTIONS_DISABLED
                try {
#endif /* BI_COMPUTE_EXCEPTIONS_ENABLED */
                    process_workloads(*_workloads, *_feeder, _info);

#ifndef BI_COMPUTE_EXCEPTIONS_DISABLED
                } catch (...) {
                    _current_exception = std::current_exception();
                }
#endif /* BI_COMPUTE_EXCEPTIONS_DISABLED */
                _workloads = nullptr;
                _job_complete = true;
                lock.unlock();
                _cv.notify_one();
            }
        }
    }

    struct BICPPScheduler::Impl final {
        constexpr static unsigned int m_default_wake_fanout = 4;

        enum class Mode {
            Linear,
            Fanout
        };

        enum class ModeToggle {
            None,
            Linear,
            Fanout
        };

        explicit Impl(unsigned int thread_hint)
            : _num_threads(thread_hint), _threads(_num_threads - 1), _mode(Mode::Linear), _wake_fanout(0U) {
            const auto mode_env_v = misc::utility::tolower(misc::utility::getenv("BI_COMPUTE_CPP_SCHEDULER_MODE"));
            if (mode_env_v == "linear") {
                _forced_mode = ModeToggle::Linear;
            } else if (mode_env_v == "fanout") {
                _forced_mode = ModeToggle::Fanout;
            } else {
                _forced_mode = ModeToggle::None;
            }
        }

        void set_num_threads(unsigned int num_threads, unsigned int thread_hint) {
            _num_threads = num_threads == 0 ? thread_hint : num_threads;
            _threads.resize(_num_threads - 1);
            auto_switch_mode(_num_threads);
        }

        void set_num_threads_with_affinity(unsigned int num_threads, unsigned int thread_hint, BindFunc func) {
            _num_threads = num_threads == 0 ? thread_hint : num_threads;

            // Set affinity on main thread
            set_thread_affinity(func(0, thread_hint));

            // Set affinity on worked threads
            _threads.clear();
            for (auto i = 1U; i < _num_threads; ++i) {
                _threads.emplace_back(func(i, thread_hint));
            }
            auto_switch_mode(_num_threads);
        }

        void auto_switch_mode(unsigned int num_threads_to_use) {
            // If the environment variable is set to any of the modes, it overwrites the mode selected over num_threads_to_use
            if (_forced_mode == ModeToggle::Fanout || (_forced_mode == ModeToggle::None && num_threads_to_use > 8)) {
                set_fanout_mode(m_default_wake_fanout, num_threads_to_use);
                BI_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(
                    "Set CPPScheduler to Fanout mode, with wake up fanout : %d and %d threads to use\n",
                    this->wake_fanout(), num_threads_to_use);
            } else
            // Equivalent to (_forced_mode == ModeToggle::Linear || (_forced_mode == ModeToggle::None && num_threads_to_use <= 8))
            {
                set_linear_mode();
                BI_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Set CPPScheduler to Linear mode, with %d threads to use\n",
                                                         num_threads_to_use);
            }
        }

        void set_linear_mode() {
            for (auto &thread: _threads) {
                thread.set_linear_mode();
            }
            _mode = Mode::Linear;
            _wake_fanout = 0U;
        }

        void set_fanout_mode(unsigned int wake_fanout, unsigned int num_threads_to_use) {
            BI_COMPUTE_ERROR_ON(num_threads_to_use > _threads.size() + 1);
            const auto actual_wake_fanout = std::max(2U, std::min(wake_fanout, num_threads_to_use - 1));
            auto thread_it = _threads.begin();
            for (auto i = 1U; i < num_threads_to_use; ++i, ++thread_it) {
                const auto wake_begin = i * actual_wake_fanout - 1;
                const auto wake_end = std::min((i + 1) * actual_wake_fanout - 1, num_threads_to_use - 1);
                thread_it->set_fanout_mode(&_threads, wake_begin, wake_end);
            }
            // Reset the remaining threads's wake up schedule
            while (thread_it != _threads.end()) {
                thread_it->set_fanout_mode(&_threads, 0U, 0U);
                ++thread_it;
            }
            _mode = Mode::Fanout;
            _wake_fanout = actual_wake_fanout;
        }

        unsigned int num_threads() const {
            return _num_threads;
        }

        unsigned int wake_fanout() const {
            return _wake_fanout;
        }

        Mode mode() const {
            return _mode;
        }

        void run_workloads(std::vector<BIIScheduler::BIWorkload> &workloads);

        unsigned int _num_threads;
        std::list<BIThread> _threads;
        BatmanInfer::Mutex _run_workloads_mutex{};
        Mode _mode{Mode::Linear};
        ModeToggle _forced_mode{ModeToggle::None};
        unsigned int _wake_fanout{0};
    };

    BICPPScheduler &BICPPScheduler::get() {
        static BICPPScheduler scheduler;
        return scheduler;
    }

    BICPPScheduler::BICPPScheduler() : _impl(std::make_unique<Impl>(num_threads_hint())) {
    }

    BICPPScheduler::~BICPPScheduler() = default;

    void BICPPScheduler::set_num_threads(unsigned int num_threads) {
        // No changes in the number of threads while current workloads are running
        BatmanInfer::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
        _impl->set_num_threads(num_threads, num_threads_hint());
    }

    void
    BICPPScheduler::set_num_threads_with_affinity(unsigned int num_threads, BatmanInfer::BIIScheduler::BindFunc func) {
        // No changes in the number of threads while current workloads are running
        BatmanInfer::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
        _impl->set_num_threads_with_affinity(num_threads, num_threads_hint(), func);
    }

    unsigned int BICPPScheduler::num_threads() const {
        return _impl->num_threads();
    }

#ifndef DOXYGEN_SKIP_THIS

    void BICPPScheduler::run_workloads(std::vector<BIWorkload> &workloads) {
        // Mutex to ensure other threads won't interfere with the setup of the current thread's workloads
        // Other thread's workloads will be scheduled after the current thread's workloads have finished
        // This is not great because different threads workloads won't run in parallel but at least they
        // won't interfere each other and deadlock.
        BatmanInfer::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
        const unsigned int num_threads_to_use = std::min(_impl->num_threads(),
                                                         static_cast<unsigned int>(workloads.size()));
        if (num_threads_to_use < 1) {
            return;
        }
        // Re-adjust the mode if the actual number of threads to use is different from the number of threads created
        _impl->auto_switch_mode(num_threads_to_use);
        int num_threads_to_start = 0;
        switch (_impl->mode()) {
            case BICPPScheduler::Impl::Mode::Fanout: {
                num_threads_to_start = static_cast<int>(_impl->wake_fanout()) - 1;
                break;
            }
            case BICPPScheduler::Impl::Mode::Linear:
            default: {
                num_threads_to_start = static_cast<int>(num_threads_to_use) - 1;
                break;
            }
        }
        BIThreadFeeder feeder(num_threads_to_use, workloads.size());
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        info.num_threads = num_threads_to_use;
        unsigned int t = 0;
        auto thread_it = _impl->_threads.begin();
        // Set num_threads_to_use - 1 workloads to the threads as the remaining 1 is left to the main thread
        for (; t < num_threads_to_use - 1; ++t, ++thread_it) {
            info.thread_id = t;
            thread_it->set_workload(&workloads, feeder, info);
        }
        thread_it = _impl->_threads.begin();
        for (int i = 0; i < num_threads_to_start; ++i, ++thread_it) {
            thread_it->start();
        }
        info.thread_id = t; // Set main thread's thread_id
        std::exception_ptr last_exception = nullptr;
#ifndef BI_COMPUTE_EXCEPTIONS_DISABLED
        try {
#endif                                              /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
            process_workloads(workloads, feeder, info); // Main thread processes workloads
#ifndef BI_COMPUTE_EXCEPTIONS_DISABLED
        } catch (...) {
            last_exception = std::current_exception();
        }

        try {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
            thread_it = _impl->_threads.begin();
            for (unsigned int i = 0; i < num_threads_to_use - 1; ++i, ++thread_it) {
                std::exception_ptr current_exception = thread_it->wait();
                if (current_exception) {
                    last_exception = current_exception;
                }
            }
            if (last_exception) {
                std::rethrow_exception(last_exception);
            }
#ifndef BI_COMPUTE_EXCEPTIONS_DISABLED
        } catch (const std::system_error &e) {
            std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
        }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
    }

#endif /* DOXYGEN_SKIP_THIS */

    void BICPPScheduler::schedule_op(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints,
                                     const BatmanInfer::BIWindow &window, BatmanInfer::BIITensorPack &tensors) {
        schedule_common(kernel, hints, window, tensors);
    }

    void BICPPScheduler::schedule(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints) {
        BIITensorPack tensors;
        schedule_common(kernel, hints, kernel->window(), tensors);
    }

    void BICPPScheduler::schedule_kv_split(BIITensorPack &tensors, const std::vector<size_t> &ava_len) {
        BI_COMPUTE_UNUSED(tensors);
    }

    void BICPPScheduler::schedule_kv_full_fill(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst, const std::vector<size_t> &ava_len) {
        BI_COMPUTE_UNUSED(tensors);
        BI_COMPUTE_UNUSED(mem_lst);
        BI_COMPUTE_UNUSED(ava_len);
    }

    void BICPPScheduler::schedule_change_q(BIITensorPack &tensors, const std::vector<size_t> &ava_len, size_t max_seq_len) {
        BI_COMPUTE_UNUSED(tensors);
        BI_COMPUTE_UNUSED(ava_len);
        BI_COMPUTE_UNUSED(max_seq_len);
    }

    void BICPPScheduler::schedule_kv_concat(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst, const std::vector<size_t> &ava_len) {
        BI_COMPUTE_UNUSED(tensors);
        BI_COMPUTE_UNUSED(mem_lst);
        BI_COMPUTE_UNUSED(ava_len);
    }
}
