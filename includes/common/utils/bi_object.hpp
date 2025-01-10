//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_COMMON_OBJECT_HPP
#define BATMANINFER_BI_COMMON_OBJECT_HPP

#include <cstdint>

namespace BatmanInfer {
    class BIIContext;

    namespace detail {
        /**< Object type enumerations */
        enum class BIObjectType : uint32_t {
            Context = 1,
            Queue = 2,
            Tensor = 3,
            TensorPack = 4,
            Operator = 5,
            Invalid = 0x56DEAD78
        };

        /**< 所有不透明结构都使用的 API 头部元数据结构 */
        struct BIHeader {
            /** Constructor
             *
             * @param[in] type_ Object identification type
             * @param[in] ctx_  Context to reference
             */
            BIHeader(BIObjectType type_, BIIContext *ctx_) noexcept: type(type_), ctx(ctx_) {
            }

            BIObjectType type{BIObjectType::Invalid};
            BIIContext *ctx{nullptr};
        };
    }
}

#endif //BATMANINFER_BI_COMMON_OBJECT_HPP
