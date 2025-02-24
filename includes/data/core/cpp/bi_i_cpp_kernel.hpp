//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_CPP_KERNEL_HPP
#define BATMANINFER_BI_I_CPP_KERNEL_HPP

#include <data/core/cpp/cpp_types.hpp>
#include <data/core/experimental/types.hpp>
#include <data/core/bi_i_kernel.hpp>
#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    class BIWindow;

    class BIITensor;

    /**
     * @brief 共同接口: 所有内核的C++实现
     */
    class BIICPPKernel : public BIIKernel {
    public:
        /**
         * @brief 默认最小工作负载大小值 - 无影响(minimum workload)
         */
        static constexpr size_t default_mws = 1;

        virtual ~BIICPPKernel() = default;

        /**
         * @brief 在传入的窗口上执行内核
         *
         * @warning 如果 is_parallelisable() 返回 false，则传入的窗口必须等于 window()。
         *
         * @note 窗口的宽度必须是 num_elems_processed_per_iteration() 的倍数。
         *
         * @param window 要执行内核的区域。（必须是由 window() 返回的窗口的一个区域）
         * @param info  关于执行线程和 CPU 的信息。
         */
        virtual void run(const BIWindow &window, const ThreadInfo &info) {
            BI_COMPUTE_UNUSED(window, info);
            BI_COMPUTE_ERROR("default implementation of legacy run() virtual member function invoked");
        }

        /**
         * @brief 为不支持 thread_locator 的实现提供的遗留兼容层
         *        在这些情况下，我们只是将接口缩减为遗留版本。
         *
         * @param window 要执行内核的区域。（必须是由 window() 返回的窗口的一个区域）
         * @param info  关于执行线程和 CPU 的信息。
         * @param thread_locator 指定当前线程在多维空间中的“位置”。
         */
        virtual void run_nd(const BIWindow &window, const ThreadInfo &info, const BIWindow &thread_locator) {
            BI_COMPUTE_UNUSED(thread_locator);
            run(window, info);
        }

        /**
         * @brief 在传入的窗口上执行内核
         *
         * @warning 如果 is_parallelisable() 返回 false，则传入的窗口必须等于 window()。
         *
         * @note 传入的窗口必须是由 window() 方法返回的窗口中的一个区域。
         *
         * @note 窗口的宽度必须是 num_elems_processed_per_iteration() 的倍数。
         *
         * @param tensors 包含要操作的张量的向量。
         * @param window  要执行内核的区域。（必须是由 window() 返回的窗口的一个区域）
         * @param info   关于执行线程和 CPU 的信息。
         */
        virtual void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
            BI_COMPUTE_UNUSED(tensors, window, info);
        }

        /**
         * @brief 返回相关内核的最小工作负载大小
         * @param platform 用于创建上下文的 CPU 平台。
         * @param thread_count 执行中的线程数量。
         * @return
         */
        virtual size_t get_mws(const CPUInfo &platform, size_t thread_count) const {
            BI_COMPUTE_UNUSED(platform, thread_count);

            return default_mws;
        }

        /**
         * 动态设置dynamic_window
         * @param window
         */
        virtual void dynamic_window(const BIWindow &window) {
            BI_COMPUTE_UNUSED(window);
        }

        /**
         * @brief 内核的名称
         * @return
         */
        virtual const char *name() const = 0;
    };
}

#endif //BATMANINFER_BI_I_CPP_KERNEL_HPP
