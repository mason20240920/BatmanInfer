//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_BI_CPU_RESHAPE_HPP
#define BATMANINFER_BI_CPU_RESHAPE_HPP

#include <data/core/bi_window.hpp>
#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * 基本的运行函数: 运行 @ref kernels::BICpuReshapeKernel 内核
         */
        class BICpuReshape : public BIICpuOperator {
        public:
            /**
             * 根据参数配置算子
             * @param src Source tensor info. Data type supported: All
             * @param dst Destination info. Data type supported: Same as @p src
             */
            void configure(const BIITensorInfo *src, BIITensorInfo *dst);

            /**
             * Static function to check if given info will lead to a valid configuration
             * @param src
             * @param dst
             * @return
             */
            static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst);

            void dynamic_configure(const BIITensorInfo *dst);

            void run(BatmanInfer::BIITensorPack &tensors) override;

        private:
            bool _is_prepared{false};
        };
    }
}

#endif //BATMANINFER_BI_CPU_RESHAPE_HPP
