//
// Created by Mason on 2025/1/16.
//

#include <runtime/neon/functions/bi_ne_mat_mul.hpp>

#include <data/core/bi_vlidate.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/operators/bi_cpu_mat_mul.hpp>
#include <utility>
#include <function_info/bi_MatMulInfo.h>

namespace BatmanInfer {
    struct BINEMatMul::Impl {
        Impl(std::shared_ptr<BIIMemoryManager> memory_manager) : memory_group(std::move(memory_manager)) {
        }

        Impl(const Impl &) = delete;

        Impl &operator=(const Impl &) = delete;

        const BIITensor *lhs{nullptr};
        const BIITensor *rhs{nullptr};
        BIITensor *output{nullptr};
        std::unique_ptr<cpu::BICpuMatMul> op{nullptr};
        BIMemoryGroup memory_group;
        WorkspaceData<BITensor> workspace_tensors{};
        BIITensorPack run_pack{};
    };

    BINEMatMul::BINEMatMul(std::shared_ptr<BIIMemoryManager> memory_manager) : _impl(
            std::make_unique<Impl>(memory_manager)) {
    }

    BINEMatMul::~BINEMatMul() = default;

    void BINEMatMul::configure(BatmanInfer::BIITensor *lhs,
                               BatmanInfer::BIITensor *rhs,
                               BatmanInfer::BIITensor *output,
                               const BatmanInfer::BIMatMulInfo &info,
                               const BatmanInfer::BICpuMatMulSettings &settings,
                               const BatmanInfer::BIActivationLayerInfo &act_info) {
        _impl->lhs = lhs;
        _impl->rhs = rhs;
        _impl->output = output;

        BI_COMPUTE_ERROR_ON_NULLPTR(_impl->lhs, _impl->rhs, _impl->output);
        _impl->op = std::make_unique<cpu::BICpuMatMul>();
        _impl->op->configure(lhs->info(), rhs->info(), output->info(), info, settings, act_info);
        _impl->run_pack = {{ACL_SRC_0, lhs},
                           {ACL_SRC_1, rhs},
                           {ACL_DST,   output}};
        _impl->workspace_tensors = manage_workspace<BITensor>(_impl->op->workspace(), _impl->memory_group,
                                                              _impl->run_pack);
    }

    BIStatus BINEMatMul::validate(const BIITensorInfo *lhs,
                                  const BIITensorInfo *rhs,
                                  const BIITensorInfo *output,
                                  const BIMatMulInfo &info,
                                  const BICpuMatMulSettings &settings,
                                  const BIActivationLayerInfo &act_info) {
        return cpu::BICpuMatMul::validate(lhs, rhs, output, info, settings, act_info);
    }

    void BINEMatMul::run() {
        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        _impl->op->run(_impl->run_pack);
    }
}