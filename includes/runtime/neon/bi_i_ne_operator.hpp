//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_NE_OPERATOR_HPP
#define BATMANINFER_BI_I_NE_OPERATOR_HPP

#include <runtime/bi_i_operator.hpp>
#include <runtime/bi_i_runtime_context.hpp>

namespace BatmanInfer {
    class BIICPPKernel;

    class BIWindow;

    using BIINEKernel = BIICPPKernel;

    namespace experimental {
        /**
         * @brief 具有单个异步 CPU 内核的函数的基本接口
         */
        class BIINEOperator : public BIIOperator {
        public:
            /**
             * @brief 构造函数
             * @param ctx
             */
            BIINEOperator(BIIRuntimeContext *ctx = nullptr);

            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BIINEOperator(const BIINEOperator &) = delete;

            BIINEOperator(BIINEOperator &&) = default;

            BIINEOperator &operator=(const BIINEOperator &) = delete;

            BIINEOperator &operator=(BIINEOperator &&) = default;

            ~BIINEOperator();

            void run(BatmanInfer::BIITensorPack &tensors) override;

            void prepare(BatmanInfer::BIITensorPack &constants) override;

            BIMemoryRequirements workspace() const override;

        protected:
            void run(BIITensorPack &tensors, const BIWindow &window);

            std::unique_ptr<BIINEKernel> _kernel;
            BIIRuntimeContext *_ctx;
            BIMemoryRequirements _workspace;

        };
    }
}

#endif //BATMANINFER_BI_I_NE_OPERATOR_HPP
