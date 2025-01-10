//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_COMMON_BI_I_OPERATOR_HPP
#define BATMANINFER_COMMON_BI_I_OPERATOR_HPP

#include <common/bi_i_context.hpp>
#include <common/bi_i_queue.hpp>

#include <data/core/experimental/types.hpp>
#include <runtime/bi_i_operator.hpp>

#include <common/utils/bi_validate.hpp>

#include <vector>

struct BclOperator_ {
    BatmanInfer::detail::BIHeader header{BatmanInfer::detail::BIObjectType::Operator, nullptr};

protected:
    BclOperator_() = default;

    ~BclOperator_() = default;
};

namespace BatmanInfer {
    // 前向声明
    class BIITensorPack;
    namespace experimental {
        class BIIOperator;
    } // namespace experimental

    using MemoryRequirements = experimental::BIMemoryRequirements;

    /**
     * 指定算子接口的基类
     */
    class BIIOperator : public BclOperator_ {
    public:
        /**
         *
         * 显示初始化算子
         *
         * @param ctx
         */
        explicit BIIOperator(BIIContext *ctx);

        virtual ~BIIOperator();

        /**
         * 验证算子是否合法
         * @return
         */
        bool is_valid() const;

        /**
         * 函数内的执行内核
         * @param queue 使用队列
         * @param tensors 包含张量的向量去运行
         * @return
         */
        virtual StatusCode run(BIIQueue &queue, BIITensorPack &tensors);

        /**
         * 运行函数里的内核
         * @param tensors
         * @return
         */
        virtual StatusCode run(BIITensorPack &tensors);

        /**
         * 为运算符准备执行
         *
         * 任何函数所需的一次性预处理步骤都在此处处理。
         *
         * @param tensors 包含准备阶段所需张量的向量。
         *
         * @note 准备阶段可能不需要函数的所有缓冲区的后备内存都可用即可执行。
         * @return
         */
        virtual StatusCode prepare(BIITensorPack &tensors);

        /**
         * 返回工作区域需要的内存
         * @return
         */
        virtual MemoryRequirements workspace() const;

        void set_internal_operator(std::unique_ptr<experimental::BIIOperator> op) {
            _op = std::move(op);
        }

    private:
        std::unique_ptr<experimental::BIIOperator> _op{nullptr};
    };

    /**
     * 提取算子的内部表示
     * @param op
     * @return
     */
    inline BIIOperator *get_internal(BclOperator op) {
        return static_cast<BIIOperator *>(op);
    }

    namespace detail {
        /**
         * 验证内部算子是否合法
         * @param op
         * @return
         */
        inline StatusCode validate_internal_operator(const BIIOperator *op) {
            if (op == nullptr || !op->is_valid()) {
                BI_COMPUTE_LOG_ERROR_ACL("[IOperator]: Invalid operator object");
                return StatusCode::InvalidArgument;
            }
            return StatusCode::Success;
        }
    }
}


#endif //BATMANINFER_COMMON_BI_I_OPERATOR_HPP
