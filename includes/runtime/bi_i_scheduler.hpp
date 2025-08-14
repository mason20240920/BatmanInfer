//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_I_SCHEDULER_HPP
#define BATMANINFER_BI_I_SCHEDULER_HPP

#include <data/core/cpp/cpp_types.hpp>
#include <data/core/experimental/types.hpp>
#include <data/core/bi_types.hpp>

#include <functional>
#include <limits>

namespace BatmanInfer {
    struct PhysicalBlock;
    // 前向声明
    class BIICPPKernel;

    class BIWindow;

    /**
     * @brief 用于运行内核的调度器接口
     */
    class BIIScheduler {
    public:
        enum class BIStrategyHint {
            /**
             * @brief 将工作负载均匀分配到各个线程中。
             */
            STATIC,
            /**
             * @brief 使用桶系统动态分配工作负载
             */
            DYNAMIC,
        };

        /**
         * @brief 用于将给定的线程 ID 映射到逻辑核心 ID 的函数
         *
         * 映射函数需要线程索引和核心总数作为输入，
         * 并返回要绑定的逻辑核心索引
         *
         * 线程 ID：表示线程的标识（通常是一个索引值）
         * 核心总数：逻辑核心的数量（比如 CPU 上的核心数）
         * 逻辑核心索引：返回值，表示线程应该绑定到的核心。
         *
         * BindFunc round_robin = [](int thread_id, int core_count) -> int {
         *     return thread_id % core_count; // 线程 ID 按核心总数取模
         *  };
         *  如果有 8 个逻辑核心（core_count = 8），线程索引为 0、1、2、... 的线程会被依次映射到核心 0、1、2、...，然后循环。
         *
         */
        using BindFunc = std::function<int(int, int)>;

        /**
         * @brief 当IScheduler::Hints::_split_dimension 被初始化为该值时，
         *        调度器可以自由地在尽可能多的维度上划分问题空间
         */
        static constexpr unsigned int split_dimensions_all = std::numeric_limits<unsigned>::max();

        /**
         * @brief 策略提示
         *
         * 由函数设置的关于如何分配给定工作负载的偏好集合
         */
        class Hints {
        public:
            /**
             * @brief 构造函数
             * @param split_dimension 用于划分内核执行窗口的维度
             * @param strategy 切分策略
             * @param threshold 动态调度封顶阈值
             */
            Hints(unsigned int split_dimension,
                  BIStrategyHint strategy = BIStrategyHint::STATIC,
                  int threshold = 0) : _split_dimension(split_dimension), _strategy(strategy),
                                       _threshold(threshold) {
            }

            /**
             * @brief 设置切分维度提示
             * @param split_dimension 用于划分内核执行窗口的维度
             * @return
             */
            Hints &set_split_dimensions(unsigned int split_dimension) {
                _split_dimension = split_dimension;
                return *this;
            }

            /**
             * @brief 返回偏好的切分维度
             * @return 切分维度
             */
            unsigned int split_dimension() const {
                return _split_dimension;
            }

            /**
             * @brief 设置策略偏好
             * @param strategy
             * @return
             */
            Hints &set_strategy(BIStrategyHint strategy) {
                _strategy = strategy;
                return *this;
            }

            BIStrategyHint strategy() const {
                return _strategy;
            }

            /**
             * @brief 返回用于动态调度的颗粒封顶阈值。
             * @return 封顶阈值
             */
            int threshold() const {
                return _threshold;
            }

        private:
            unsigned int _split_dimension{};
            // 负载策略
            BIStrategyHint _strategy{};
            /**
             * @brief 限制动态调度时任务颗粒的大小
             * 颗粒度：在动态调度中，问题空间被划分为多个小任务（颗粒）。每个颗粒是一个基本的计算单元。
             * 阈值的意义：_threshold 的值决定了每个线程在动态调度中一次性分配的任务量上限。
             *      1. 如果 threshold 值较小，则颗粒度较小，任务划分更细，调度更加灵活，但调度开销可能增加。
             *      2. 如果 threshold 值较大，则颗粒度较大，调度开销减少，但可能导致负载不均衡。
             */
            int _threshold{};
        };

        /**
         * @brief 用于执行工作负载的签名
         */
        using BIWorkload = std::function<void(const ThreadInfo &)>;

        BIIScheduler();

        virtual ~BIIScheduler() = default;

        /**
         * @brief 设置调度程序用于运行内核的线程数。
         * @param num_threads 如果设置为0，则每个系统可用的CPU核心将使用一个线程，否则将使用指定数量的线程。
         */
        virtual void set_num_threads(unsigned int num_threads) = 0;

        /**
         * @brief 设置调度器用于运行内核的线程数量，同时使用绑定函数将线程固定到指定的`逻辑核心`上。
         * @param num_threads 如果设置为0，则每个系统可用的CPU核心将使用一个线程，否则将使用指定数量的线程。
         * @param func 绑定函数去使用
         */
        virtual void set_num_threads_with_affinity(unsigned int num_threads, BindFunc func);

        /**
         * @brief 返回 SingleThreadScheduler 池中线程的数量。
         * @return SingleThreadScheduler 中可用的线程数量。
         */
        virtual unsigned int num_threads() const = 0;

        /**
         * @brief 在与调用者相同的线程中同步运行内核。
         *
         * @param kernel 要执行的内核。
         * @param hints 给调度程序的提示。
         */
        virtual void schedule(BIICPPKernel *kernel, const Hints &hints) = 0;

        /**
         * @brief 在与调用者相同的线程中同步运行内核。
         * @param kernel  要执行的内核。
         * @param hints 给调度程序的提示。
         * @param window
         * @param tensors 包含要操作的张量的向量。
         */
        virtual void
        schedule_op(BIICPPKernel *kernel, const Hints &hints, const BIWindow &window, BIITensorPack &tensors) = 0;

        /**
         * @brief 调度kv cache的切分拷贝
         * @param tensors 张量信息
         * @param ava_len 可用长度
         */
        virtual void schedule_kv_split(BIITensorPack &tensors, const std::vector<size_t>& ava_len) = 0;

        /**
         * @brief 调度KV Cache Manager的合并接口
         * @param tensors 张量信息
         * @param mem_lst 内存块数组 - 默认内存块大小(N, num heads, sequence length, head dim)
         * @param ava_len 有效长度 - 有效长度: 后面需要填充进去
         * @param layer_idx 层的索引
         */
        virtual void schedule_kv_concat(BIITensorPack &tensors,
                                        const std::vector<PhysicalBlock *> &mem_lst,
                                        const std::vector<size_t> &ava_len,
                                        int layer_idx) = 0;

        /**
         * @brief 填充Tensor中有eos部分的张量
         *
         * @param tensors
         * @param mem_lst
         * @param ava_len
         */
        virtual void schedule_kv_full_fill(BIITensorPack &tensors, const std::vector<PhysicalBlock *>&mem_lst, const std::vector<size_t> &ava_len) = 0;

        /**
         * @brief 调度修改Q矩阵
         *
         * @param tensors
         * @param ava_len
         */
        virtual void schedule_change_q(BIITensorPack &tensors, const std::vector<size_t> &ava_len, const size_t max_seq_len) = 0;

        /**
         * @brief 执行所有传递的工作负载
         *
         * @note 无法保证工作负载的执行顺序，也无法保证它们是否会并行执行。
         *
         * @param workloads 要运行的工作负载列表
         * @param tag 可供分析工具使用的字符串，用于识别由调度器运行的工作负载（可以为 null）。
         */
        virtual void run_tagged_workloads(std::vector<BIWorkload> &workloads,
                                          const char *tag);

        /**
         * @brief 获取CPU信息
         * @return
         */
        CPUInfo &cpu_info();

        /**
         * @brief 获取最佳可能的执行线程数量的提示
         * @warning 如果无法计算出最佳线程数量，
         *          则返回 std::thread::hardware_concurrency()；对于裸机构建，则返回 1
         * @return  最佳可能的执行线程数量
         */
        unsigned int num_threads_hint() const;

    protected:
        /** 执行所有传入的工作负载
        *
        * @note 不保证工作负载的执行顺序，也不保证它们是否会并行执行。
        *
        * @param[in] workloads 要运行的工作负载数组
        */
        virtual void run_workloads(std::vector<BIWorkload> &workloads) = 0;

        /** 调度器执行给定内核的通用逻辑
         *
         * @param[in] kernel  要执行的内核。
         * @param[in] hints   调度器的提示。
         * @param[in] window  用于内核执行的窗口。
         * @param[in] tensors 包含操作张量的向量。
         */
        void schedule_common(BIICPPKernel *kernel, const Hints &hints, const BIWindow &window, BIITensorPack &tensors);

        /** 调整窗口数量以优化性能
         * （用于小型工作负载，其中较少的线程数量可能会提高性能）
         *
         * @param[in] window           用于内核执行的窗口
         * @param[in] split_dimension  要划分的维度轴
         * @param[in] init_num_windows 初始划分的子窗口数量
         * @param[in] kernel           要执行的内核
         * @param[in] cpu_info         用于创建上下文的 CPU 平台信息
         *
         * @return 调整后的窗口数量
         */
        std::size_t adjust_num_of_windows(const BIWindow &window,
                                          std::size_t split_dimension,
                                          std::size_t init_num_windows,
                                          const BIICPPKernel &kernel,
                                          const CPUInfo &cpu_info);

    private:
        /**
         * @brief 线程数提示
         */
        unsigned int _num_threads_hint = {};
    };
}

#endif //BATMANINFER_BI_I_SCHEDULER_HPP
