//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_I_FUNCTION_HPP
#define BATMANINFER_BI_I_FUNCTION_HPP

namespace BatmanInfer {
    /**
     * @brief 所有函数的接口类
     */
    class BIIFunction {
    public:
        /**
         * @brief 默认析构函数
         */
        virtual ~BIIFunction() = default;

        /**
         * @brief 运行函数中包含的内核
         * 对于 CPU 内核：
         *   - 对于可并行化的内核，会使用多线程。
         *   - 默认情况下，使用 std::thread::hardware_concurrency() 返回的线程数量。
         *
         *   @note 可以通过 @ref CPPScheduler::set_num_threads() 手动设置线程数量。
         *
         * 对于 OpenCL 内核：
         *    - 所有内核都会被加入到与 CLScheduler 关联的队列中。
         *    - 队列会被刷新（flush）。
         *
         *   @note 此函数不会阻塞，直到内核执行完成。用户需要自行负责等待内核执行完成。
         *   @note 如果尚未调用 prepare()，将在首次运行时调用该方法。
         *
         */
        // virtual BIErrCode run() = 0;

        /** 为执行准备函数
         *
         * 函数所需的任何一次性预处理步骤都会在这里处理。
         *
         * @note 准备阶段可能不需要函数的所有缓冲区的后备内存都可用即可执行。
         */
        virtual void prepare() {

        }
    };
}

#endif //BATMANINFER_BI_I_FUNCTION_HPP
