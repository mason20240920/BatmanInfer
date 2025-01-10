//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_I_TENSOR_V2_HPP
#define BATMANINFER_BI_I_TENSOR_V2_HPP

#include <common/bi_i_context.hpp>
#include <common/utils/bi_validate.hpp>

struct BclTensor_ {
    BatmanInfer::detail::BIHeader header{BatmanInfer::detail::BIObjectType::Tensor, nullptr};

protected:
    BclTensor_() = default;

    ~BclTensor_() = default;
};

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BIITensorV2 : public BclTensor_ {
    public:
        /**
         * 显式算子初始化
         * @param ctx
         */
        explicit BIITensorV2(BIIContext *ctx) : BclTensor_() {
            BI_COMPUTE_ASSERT_NOT_NULLPTR(ctx);
            this->header.ctx = ctx;
            this->header.ctx->inc_ref();
        }

        virtual ~BIITensorV2() {
            this->header.ctx->dec_ref();
            this->header.type = detail::BIObjectType::Invalid;
        }

        bool is_valid() const {
            return this->header.type == detail::BIObjectType::Tensor;
        }

        /**
         * 映射张量到一个主机指针上
         * @return 如果成功，则指向底层支持内存的指针；否则为 nullptr。
         */
        virtual void *map() = 0;

        /**
         * 取消映射张量
         * @return
         */
        virtual StatusCode unmap() = 0;

        /**
         * 导入额外的内存处理
         * @param handle
         * @param type
         * @return
         */
        virtual StatusCode import(void *handle, ImportMemoryType type) = 0;

        /**
         * 获取合法的张量对象
         *
         *
         *
         * @return
         */
        virtual BatmanInfer::BIITensor *tensor() const = 0;

        /**
         * @note 大小不是基于已分配的内存，而是基于其描述符中的信息（维度、数据类型等）。
         *
         * @return 张量的大小（以字节为单位）
         */
        size_t get_size() const;

        /**
         * 获取张量的描述
         * @return
         */
        BclTensorDescriptor get_descriptor() const;
    };

    inline BIITensorV2 *get_internal(BclTensor tensor) {
        return static_cast<BIITensorV2 *>(tensor);
    }

    namespace detail {
        inline StatusCode validate_internal_tensor(const BIITensorV2 *tensor) {
            if (tensor == nullptr || !tensor->is_valid()) {
                BI_COMPUTE_LOG_ERROR_ACL("[ITensorV2]: Invalid tensor object");
                return StatusCode::InvalidArgument;
            }
            return StatusCode::Success;
        }
    }
}

#endif //BATMANINFER_BI_I_TENSOR_V2_HPP
