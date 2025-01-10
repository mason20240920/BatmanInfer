//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_I_CONTEXT_HPP
#define BATMANINFER_BI_I_CONTEXT_HPP

#include <common/bi_types.hpp>
#include <common/utils/bi_log.hpp>
#include <common/utils/bi_object.hpp>

#include <atomic>
#include <tuple>

struct BclContext_ {
    BatmanInfer::detail::BIHeader header{BatmanInfer::detail::BIObjectType::Context, nullptr};

protected:
    BclContext_() = default;

    ~BclContext_() = default;
};

namespace BatmanInfer {
    // Forward declarations
    class BIITensorV2;

    class BIIQueue;

    class BIIOperator;

    class BIIContext : public BclContext_ {
    public:
        BIIContext(BITarget target) : BclContext_(), _target(target), _refcount(0) {

        }

        /**
         * 虚拟析构函数
         */
        virtual ~BIIContext() {
            header.type = detail::BIObjectType::Invalid;
        }

        /**
         * 目标类型访问器
         * @return
         */
        BITarget type() const {
            return _target;
        }

        /** Increment context refcount */
        void inc_ref() const {
            ++_refcount;
        }

        /** Decrement context refcount */
        void dec_ref() const {
            --_refcount;
        }

        /** Reference counter accessor
         *
         * @return The number of references pointing to this object
         */
        int refcount() const {
            return _refcount;
        }

        /** Checks if an object is valid
         *
         * @return True if successful otherwise false
         */
        bool is_valid() const {
            return header.type == detail::BIObjectType::Context;
        }

        /** Create a tensor object
         *
         * @param[in] desc     Descriptor to use
         * @param[in] allocate Flag to allocate tensor
         *
         * @return A pointer to the created tensor object
         */
        virtual BIITensorV2 *create_tensor(const BclTensorDescriptor &desc, bool allocate) = 0;

        /** Create a queue object
         *
         * @param[in] options Queue options to be used
         *
         * @return A pointer to the created queue object
         */
        virtual BIIQueue *create_queue(const BclQueueOptions *options) = 0;

        virtual std::tuple<BIIOperator *, StatusCode> create_activation(const BclTensorDescriptor &src,
                                                                        const BclTensorDescriptor &dst,
                                                                        const BclActivationDescriptor &act,
                                                                        bool is_validate) = 0;

    private:
        /**
         * 不同的运行平台
         */
        BITarget _target;
        /**
         * 引用计数
         */
        mutable std::atomic<int> _refcount;
    };

    /** Extract internal representation of a Context
     *
     * @param[in] ctx Opaque context pointer
     *
     * @return The internal representation as an IContext
     */
    inline BIIContext *get_internal(BclContext ctx) {
        return static_cast<BIIContext *>(ctx);
    }

    namespace detail {
        /**
         * 检查内在的上下文是否合法
         * @param ctx
         * @return
         */
        inline StatusCode validate_internal_context(const BIIContext *ctx) {
            if (ctx == nullptr || !ctx->is_valid()) {
                BI_COMPUTE_LOG_ERROR_ACL("Invalid context object");
                return StatusCode::InvalidArgument;
            }
            return StatusCode::Success;
        }
    }
}


#endif //BATMANINFER_BI_I_CONTEXT_HPP
