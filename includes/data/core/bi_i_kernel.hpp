//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_KERNEL_HPP
#define BATMANINFER_BI_I_KERNEL_HPP

#include <data/core/bi_types.hpp>
#include <data/core/bi_window.hpp>

namespace BatmanInfer {
    /**
     * @brief 所有内核的共同信息
     */
    class BIIKernel {
    public:
        /**
         * @brief 构造函数
         */
        BIIKernel();

        /**
         * @brief 析构函数
         */
        virtual ~BIIKernel() = default;

        /**
         * @brief 确定内核是否是并行的
         *
         * 如果内核可以并行化，则通过 window() 返回的窗口可以被分割为子窗口，
         * 然后这些子窗口可以并行运行。
         *
         * 如果内核不支持并行化，则只能将 window() 返回的完整窗口传递给 run()。
         *
         * @return 如何内核支持并行化，返回True
         */
        virtual bool is_parallelisable() const;

        /**
         * @brief 获取该内核的边界大小
         *
         * @return 边界的宽度（以元素数量表示）。
         */
        virtual BIBorderSize border_size() const;

        /**
         * @brief 获取内核可以执行的最大窗口
         *
         * @return 内核可以执行的最大窗口。
         */
        const BIWindow &window() const;

        /**
         * @brief 检查该内核的嵌入式窗口是否已被配置
         * @return 如果窗口已被配置，则返回 True。
         */
        bool is_window_configured() const;

    protected:
        /** Configure the kernel's window
     *
     * @param[in] window The maximum window which will be returned by window()
     */
        void configure(const BIWindow &window);

        /**
         * 动态更新内核接口窗口
         * @param window
         */
        void dynamic_configure(const BIWindow &window);

    private:
        BIWindow _window;
    };
}

#endif //BATMANINFER_BI_I_KERNEL_HPP
