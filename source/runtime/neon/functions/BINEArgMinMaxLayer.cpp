//
// Created by Mason on 2025/4/10.
//

#include <runtime/neon/functions/BINEArgMinMaxLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/functions/bi_ne_cast.h>
#include <runtime/neon/functions/BINEReductionOperation.hpp>
#include <runtime/bi_tensor.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/BINEReductionOperationKernel.hpp>

namespace BatmanInfer {
    struct BINEArgMinMaxLayer::Impl {
        BIMemoryGroup memory_group{};
        std::shared_ptr<BIIMemoryManager> memory_manager{};
        std::unique_ptr<BINEReductionOperation> reduction_function{};
        std::unique_ptr<BINECast> cast_function{};
        std::unique_ptr<BITensor> tmp_reduction_result{};
    };

    BINEArgMinMaxLayer::~BINEArgMinMaxLayer() = default;

    BINEArgMinMaxLayer::BINEArgMinMaxLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _impl(
        std::make_unique<Impl>()) {
        _impl->memory_manager = std::move(memory_manager);
    }

    void BINEArgMinMaxLayer::configure(BIITensor *input, int axis, BIITensor *output, const BIReductionOperation &op) {
        BI_COMPUTE_LOG_PARAMS(input, axis, output, op);
        _impl->reduction_function = std::make_unique<BINEReductionOperation>();
        if (output->info() &&
            (output->info()->data_type() == BIDataType::S64 || output->info()->data_type() == BIDataType::U64)) {
            _impl->memory_group = BIMemoryGroup(std::move(_impl->memory_manager));
            _impl->cast_function = std::make_unique<BINECast>();
            _impl->tmp_reduction_result = std::make_unique<BITensor>();
            _impl->reduction_function->configure(input, _impl->tmp_reduction_result.get(), axis, op, false);
            _impl->cast_function->configure(_impl->tmp_reduction_result.get(), output, BIConvertPolicy::SATURATE);
            _impl->memory_group.manage(_impl->tmp_reduction_result.get());
            _impl->tmp_reduction_result->allocator()->allocate();
        } else {
            _impl->reduction_function->configure(input, output, axis, op, false);
        }
    }

    BIStatus
    BINEArgMinMaxLayer::validate(const BIITensorInfo *input, int axis, const BIITensorInfo *output,
                                 const BIReductionOperation &op) {
        BI_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output);
        BI_COMPUTE_RETURN_ERROR_ON_MSG(
            op != BIReductionOperation::ARG_IDX_MAX && op != BIReductionOperation::ARG_IDX_MIN,
            "Invalid operation");
        return BINEReductionOperation::validate(input, output, axis, op, false);
    }

    void BINEArgMinMaxLayer::run() {
        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        _impl->reduction_function->run();
        if (_impl->tmp_reduction_result != nullptr) {
            _impl->cast_function->run();
        }
    }
} // namespace BatmanInfer
