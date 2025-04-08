//
// Created by Mason on 2025/1/18.
//

#include <runtime/neon/functions/bi_NESoftmaxLayer.h>

#include <data/core/bi_vlidate.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include <data/core/helpers/bi_memory_helpers.hpp>
#include <data/core/helpers/bi_softmax_helpers.hpp>
#include <cpu/operators/bi_cpu_softmax.hpp>

namespace BatmanInfer {
    template<bool IS_LOG>
    struct BINESoftmaxLayerGeneric<IS_LOG>::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuSoftmaxGeneric> op{nullptr};
        BIMemoryGroup memory_group{};
        BIITensorPack run_pack{};
        WorkspaceData<BITensor> workspace_tensors{};
    };

    template<bool IS_LOG>
    BINESoftmaxLayerGeneric<IS_LOG>::BINESoftmaxLayerGeneric(std::shared_ptr<BIIMemoryManager> memory_manager)
            : _impl(std::make_unique<Impl>()) {
        _impl->memory_group = BIMemoryGroup(std::move(memory_manager));
    }

    template<bool IS_LOG>
    BINESoftmaxLayerGeneric<IS_LOG>::BINESoftmaxLayerGeneric(BINESoftmaxLayerGeneric &&) = default;

    template<bool IS_LOG>
    BINESoftmaxLayerGeneric<IS_LOG> &BINESoftmaxLayerGeneric<IS_LOG>::operator=(BINESoftmaxLayerGeneric &&) = default;

    template<bool IS_LOG>
    BINESoftmaxLayerGeneric<IS_LOG>::~BINESoftmaxLayerGeneric() = default;

    template<bool IS_LOG>
    void BINESoftmaxLayerGeneric<IS_LOG>::configure(BIITensor *input, BIITensor *output, float beta, int32_t axis) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);

        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuSoftmaxGeneric>();
        _impl->op->configure(input->info(), output->info(), beta, axis, IS_LOG);

        _impl->run_pack = {{BITensorType::ACL_SRC, _impl->src},
                           {BITensorType::ACL_DST, _impl->dst}};
        _impl->workspace_tensors = manage_workspace<BITensor>(_impl->op->workspace(), _impl->memory_group,
                                                              _impl->run_pack);
    }

    template<bool IS_LOG>
    void BINESoftmaxLayerGeneric<IS_LOG>::dynamic_configure() {
        _impl->op->dynamic_configure(_impl->src->info(), _impl->dst->info());
    }

    template<bool IS_LOG>
    BIStatus
    BINESoftmaxLayerGeneric<IS_LOG>::validate(const BIITensorInfo *input, const BIITensorInfo *output, float beta,
                                              int32_t axis) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_RETURN_ON_ERROR(cpu::BICpuSoftmaxGeneric::validate(input, output, beta, axis, IS_LOG));
        return BIStatus{};
    }

    template<bool IS_LOG>
    void BINESoftmaxLayerGeneric<IS_LOG>::run() {
        // Acquire all the temporaries
        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        BI_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);
        _impl->op->run(_impl->run_pack);
    }

    template
    class BINESoftmaxLayerGeneric<false>;

    template
    class BINESoftmaxLayerGeneric<true>;
}