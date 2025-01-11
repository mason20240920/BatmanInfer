//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BI_TENSOR_PACK_HPP
#define BATMANINFER_BI_TENSOR_PACK_HPP

#include <data/core/bi_i_tensor_pack.hpp>

#include <common/bi_i_context.hpp>

struct BclTensorPack_ {
    BatmanInfer::detail::BIHeader header{BatmanInfer::detail::BIObjectType::TensorPack, nullptr};

protected:
    BclTensorPack_() = default;

    ~BclTensorPack_() = default;
};

namespace BatmanInfer {
    // Forward declaration
    class BIITensor;

    class BIITensorV2;

    /**
     * 张量打包服务
     *
     * 类负责创建和管理张量集合。
     * 张量包可以传递给运算符，成为执行的可变数据的组成部分
    */

    class BITensorPack : public BclTensorPack_ {
    public:
        /** Constructor
         *
         * @param[in] ctx Context to be used
         */
        explicit BITensorPack(BIIContext *ctx);

        /** Destructor */
        ~BITensorPack();

        /** Add tensor to the pack
         *
         * @param[in] tensor  Tensor to add
         * @param[in] slot_id Slot identification in respect to the operator of the tensor to add
         *
         * @return Status code
         */
        BclStatus add_tensor(BIITensorV2 *tensor, int32_t slot_id);

        /** Pack size accessor
         *
         * @return Number of tensors registered to the pack
         */
        size_t size() const;

        /** Checks if pack is empty
         *
         * @return True if empty else false
         */
        bool empty() const;

        /** Checks if an object is valid
         *
         * @return True if valid else false
         */
        bool is_valid() const;

        /** Get tensor of a given id from the pac
         *
         * @param[in] slot_id Slot identification of tensor to extract
         *
         * @return The pointer to the tensor if exist and is non-const else nullptr
         */
        BatmanInfer::BIITensor *get_tensor(int32_t slot_id);

        /** Get legacy tensor pack
         *
         * @return Legacy tensor pack
         */
        BatmanInfer::BIITensorPack &get_tensor_pack();

    private:
        BatmanInfer::BIITensorPack _pack; /**< Pack that currently redirects to the existing TensorPack */
    };

/** Extract internal representation of a TensoPack
 *
 * @param[in] pack Opaque tensor pack pointer
 *
 * @return The internal representation as an TensorPack
 */
    inline BITensorPack *get_internal(BclTensorPack pack) {
        return static_cast<BITensorPack *>(pack);
    }

    namespace detail {
/** Check if an internal TensorPack is valid
 *
 * @param[in] pack Internal tensor pack to check
 *
 * @return A status code
 */
        inline StatusCode validate_internal_pack(const BITensorPack *pack) {
            if (pack == nullptr || !pack->is_valid()) {
                BI_COMPUTE_LOG_ERROR_ACL("[TensorPack]: Invalid tensor pack object");
                return StatusCode::InvalidArgument;
            }
            return StatusCode::Success;
        }
    } // namespace detail
}

#endif //BATMANINFER_BI_TENSOR_PACK_HPP
