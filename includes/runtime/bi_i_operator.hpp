//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_OPERATOR_HPP
#define BATMANINFER_BI_I_OPERATOR_HPP

#include <data/core/experimental/types.hpp>
#include <runtime/bi_i_runtime_context.hpp>
#include <runtime/bi_types.hpp>

namespace BatmanInfer {
    namespace experimental {
        /**
         * @brief 所有异步函数的基类
         */
        class BIIOperator {
        public:
            /**
             * @brief 析构函数
             */
            virtual ~BIIOperator() = default;

            /**
             * @brief 运行在函数内部的核心
             * @param tensors 向量(包含张量)执行
             */
            virtual void run(BIITensorPack &tensors) = 0;

            /**
             * @brief 准备函数以便执行
             *
             * 任何函数执行所需的一次性预处理步骤都在此处处理。
             *
             * @param constants 包含常量张量的向量
             *
             * @note 准备阶段可能不需要函数所有缓冲区的后备内存可用即可执行。
             */
            virtual void prepare(BIITensorPack &constants) = 0;

            /**
             * @brief 返回工作区间需要的内存(一个内存信息的数组)
             * @return
             */
            virtual BIMemoryRequirements workspace() const = 0;

            /**
             * @brief 返回动态工作区域的内存管理数组
             * @param tensors 这个集合的数组会在后面进行执行
             * @return 返回空的内存MemoryInfo数组
             */
            virtual const BIMemoryRequirements &workspace_dynamic(const BIITensorPack &tensors) const {
                BI_COMPUTE_UNUSED(tensors);
                static BIMemoryRequirements empty{};
                return empty;
            }
        };
    }
}

#endif //BATMANINFER_BI_I_OPERATOR_HPP
