//
// Created by Mason on 2025/1/11.
//

#include <cpu/bi_cpu_tensor.hpp>

#include <common/utils/legacy_support.hpp>

namespace BatmanInfer {
    namespace cpu {
        BICpuTensor::BICpuTensor(BatmanInfer::BIIContext *ctx, const BclTensorDescriptor &desc) : BIITensorV2(ctx),
                                                                                                  _legacy_tensor() {
            BI_COMPUTE_ASSERT((ctx != nullptr) && (ctx->type() == Target::Cpu));
            _legacy_tensor = std::make_unique<BITensor>();
            _legacy_tensor->allocator()->init(BatmanInfer::detail::convert_to_legacy_tensor_info(desc));
        }

        void *BICpuTensor::map() {
            BI_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

            if (_legacy_tensor == nullptr) {
                BI_COMPUTE_LOG_ERROR_ACL("[BICpuTensor:map]: Backing tensor does not exist!");
                return nullptr;
            }
            return _legacy_tensor->buffer();
        }

        StatusCode BICpuTensor::allocate() {
            BI_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

            _legacy_tensor->allocator()->allocate();
            return StatusCode::Success;
        }

        StatusCode BICpuTensor::unmap() {
            // No-op
            return StatusCode::Success;
        }

        StatusCode BICpuTensor::import(void *handle, ImportMemoryType type) {
            BI_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);
            BI_COMPUTE_UNUSED(type);

            const auto st = _legacy_tensor->allocator()->import_memory(handle);
            return bool(st) ? StatusCode::Success : StatusCode::RuntimeError;
        }

        BatmanInfer::BIITensor *BICpuTensor::tensor() const {
            return _legacy_tensor.get();
        }
    }
}