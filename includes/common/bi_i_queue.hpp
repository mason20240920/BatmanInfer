//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_COMMON_I_QUEUE_HPP
#define BATMANINFER_BI_COMMON_I_QUEUE_HPP

#include <common/bi_i_context.hpp>

struct BclQueue_ {
    BatmanInfer::detail::BIHeader header{BatmanInfer::detail::BIObjectType::Queue, nullptr};

protected:
    BclQueue_() = default;

    ~BclQueue_() = default;
};

namespace BatmanInfer {
    /**
     * 指定队列接口的基类
     */
    class BIIQueue : public BclQueue_ {
    public:
        /**
         * 显式运算符构造函数
         *
         * @param ctx 算子使用的上下文
         */
        explicit BIIQueue(BIIContext *ctx) {
            this->header.ctx = ctx;
            this->header.ctx->inc_ref();
        }

        virtual ~BIIQueue() {
            this->header.ctx->dec_ref();
            this->header.type = detail::BIObjectType::Invalid;
        }

        /**
         * 验证队列是否有效
         *
         * @return
         */
        bool is_valid() const {
            return this->header.type == detail::BIObjectType::Queue;
        }

        virtual StatusCode finish() = 0;
    };

    /**
     * 提取队列的内部表示
     * @param queue 不透明队列指针
     * @return
     */
    inline BIIQueue *get_internal(BclQueue queue) {
        return static_cast<BIIQueue *>(queue);
    }

    namespace detail {
        /**
         * 验证内部队列是否合法
         *
         * @param queue 内部队列去验证
         * @return
         */
        inline StatusCode validate_internal_queue(const BIIQueue *queue) {
            if (queue == nullptr || !queue->is_valid()) {
                BI_COMPUTE_LOG_ERROR_ACL("[IQueue]: Invalid queue object");
                return StatusCode::InvalidArgument;
            }
            return StatusCode::Success;
        }
    }
}

#endif //BATMANINFER_BI_COMMON_I_QUEUE_HPP
