//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BI_CPU_CONTEXT_HPP
#define BATMANINFER_BI_CPU_CONTEXT_HPP

#include <common/bi_allocator_wrapper.hpp>
#include <common/cpu_info/cpu_info.hpp>
#include <common/bi_i_context.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * 编码 CPU 功能的结构
         */
        struct BICpuCapabilities {
            cpu_info::CpuInfo cpu_info{};
            int32_t max_threads{-1};
        };

        class BICpuContext final : public BIIContext {
        public:
            /**
             * 默认构造函数
             * @param options
             */
            explicit BICpuContext(const BclContextOptions *options);

            /**
             * Cpu功能结构
             * @return
             */
            const BICpuCapabilities &capabilities() const;

            /**
             * 内存分配器
             * @return
             */
            BIAllocatorWrapper &allocator();

            BIITensorV2 *create_tensor(const BclTensorDescriptor &desc, bool allocate) override;

            BIIQueue *create_queue(const BclQueueOptions *options) override;

            std::tuple<BIIOperator *, StatusCode>
            create_activation(const BclTensorDescriptor &src,
                              const BclTensorDescriptor &dst,
                              const BclActivationDescriptor &act,
                              bool is_validate) override;

        private:
            /**
             * 内存分配器实现
             */
            BIAllocatorWrapper _allocator;

            /**
             * CPU功能结构
             */
            BICpuCapabilities _caps;
        };
    }
}

#endif //BATMANINFER_BI_CPU_CONTEXT_HPP
