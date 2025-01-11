//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BI_CPU_QUEUE_HPP
#define BATMANINFER_BI_CPU_QUEUE_HPP

#include <runtime/bi_i_scheduler.hpp>

#include <common/bi_i_queue.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * CPU 队列实现类
         */
        class BICpuQueue final : public BIIQueue {
        public:

            /**
             * 初始化一个CPU队列
             * @param ctx 使用的上下文
             * @param options
             */
            BICpuQueue(BIIContext *ctx,
                       const BclQueueOptions *options);

            /**
             * 运行合法调度器
             * @return
             */
            BatmanInfer::BIIScheduler &scheduler();

            StatusCode finish() override;
        };
    }
}

#endif //BATMANINFER_BI_CPU_QUEUE_HPP
