//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BI_CPU_TENSOR_HPP
#define BATMANINFER_BI_CPU_TENSOR_HPP

#include <runtime/bi_tensor.hpp>

#include <common/bi_i_tensor_v2.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * CPU张量实现类
         */
        class BICpuTensor final : public BIITensorV2 {
        public:
            /**
             * 初始化CPU张量对象
             * @param ctx
             * @param desc
             */
            BICpuTensor(BIIContext *ctx,
                        const BclTensorDescriptor &desc);

            /**
             * 分配张量内存
             * @return
             */
            StatusCode allocate();

            void *map() override;

            StatusCode unmap() override;

            BatmanInfer::BIITensor *tensor() const override;

            StatusCode import(void *handle, BatmanInfer::ImportMemoryType type) override;

        private:
            std::unique_ptr<BITensor> _legacy_tensor;
        };
    }
}

#endif //BATMANINFER_BI_CPU_TENSOR_HPP
