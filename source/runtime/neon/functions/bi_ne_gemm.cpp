//
// Created by Mason on 2025/1/14.
//

#include <runtime/neon/functions/bi_ne_gemm.hpp>

#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/operators/bi_cpu_gemm.hpp>

using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    struct BINEGEMM::Impl {
        BIMemoryGroup memory_group{};
        BIIWeightsManager *weights_manager{nullptr};

        std::unique_ptr<cpu::BICpuGemm> op{nullptr};

        const BIITensor *original_b{nullptr};
        bool is_prepared{false};

        BIITensorPack run_pack{};
        BIITensorPack prep_pack{};
        WorkspaceData<BITensor> workspace{};
        experimental::BIMemoryRequirements aux_mem_req{};
    };

    BINEGEMM::BINEGEMM(std::shared_ptr<BIIMemoryManager> memory_manager,
                       BatmanInfer::BIIWeightsManager *weights_manager) : _impl(std::make_unique<Impl>()) {
        _impl->memory_group = BIMemoryGroup(std::move(memory_manager));
        _impl->weights_manager = weights_manager;
    }

    BINEGEMM::~BINEGEMM() = default;

    void BINEGEMM::configure(const BatmanInfer::BIITensor *a, const BatmanInfer::BIITensor *b,
                             const BatmanInfer::BIITensor *c, BatmanInfer::BIITensor *d, float alpha, float beta,
                             const BatmanInfer::GEMMInfo &gemm_info) {
        BI_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
        BI_COMPUTE_ERROR_THROW_ON(cpu::BICpuGemm::validate(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr,
                                                           d->info(), alpha, beta, gemm_info));

        // Check if we need to reshape the matrix B only on the first run
        _impl->is_prepared = false;
        _impl->memory_group.mappings().clear();
        _impl->original_b = b;
        _impl->op = std::make_unique<cpu::BICpuGemm>();

        // Make the B matrix dynamic values.
        auto b_info_to_use = b->info()->clone();
        if (!gemm_info.reshape_b_only_on_first_run()) {
            b_info_to_use->set_are_values_constant(false);
        }

        _impl->op->configure(a->info(), b_info_to_use.get(), (c != nullptr) ? c->info() : nullptr, d->info(), alpha,
                             beta,
                             gemm_info);

        _impl->aux_mem_req = _impl->op->workspace();
        _impl->run_pack = {{ACL_SRC_0, a},
                           {ACL_SRC_1, b},
                           {ACL_SRC_2, c},
                           {ACL_DST,   d}};
        _impl->prep_pack = {{ACL_SRC_1, b},
                            {ACL_SRC_2, c}};
        _impl->workspace = manage_workspace<BITensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack,
                                                      _impl->prep_pack, /* allocate_now */ false);
    }

    BIStatus BINEGEMM::validate(const BatmanInfer::BIITensorInfo *a, const BatmanInfer::BIITensorInfo *b,
                                const BatmanInfer::BIITensorInfo *c, const BatmanInfer::BIITensorInfo *output,
                                float alpha, float beta, const BatmanInfer::GEMMInfo &gemm_info) {
        // Make the B matrix dynamic values.
        auto b_to_use = b->clone();
        if (!gemm_info.reshape_b_only_on_first_run()) {
            b_to_use->set_are_values_constant(false);
        }

        return cpu::BICpuGemm::validate(a, b_to_use.get(), c, output, alpha, beta, gemm_info);
    }

    BIStatus
    BINEGEMM::has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format, const BatmanInfer::BIITensorInfo *a,
                           const BatmanInfer::BIITensorInfo *b, const BatmanInfer::BIITensorInfo *c,
                           const BatmanInfer::BIITensorInfo *output, float alpha, float beta,
                           const BatmanInfer::GEMMInfo &gemm_info) {
        BI_COMPUTE_UNUSED(alpha, beta);
        return cpu::BICpuGemm::has_opt_impl(expected_weight_format, a, b, c, output, gemm_info);
    }

    void BINEGEMM::run() {
        prepare();

        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        _impl->op->run(_impl->run_pack);
    }

    void BINEGEMM::prepare() {
        if (!_impl->is_prepared) {
            allocate_tensors(_impl->aux_mem_req, _impl->workspace);
            _impl->op->prepare(_impl->prep_pack);

            auto has_reshape =
                    std::find_if(_impl->aux_mem_req.begin(), _impl->aux_mem_req.end(),
                                 [](const BIMemoryInfo &m) -> bool {
                                     return m.lifetime == MemoryLifetime::Persistent;
                                 });

            if (has_reshape != std::end(_impl->aux_mem_req)) {
                _impl->original_b->mark_as_unused();
            } else {
                _impl->run_pack.add_const_tensor(ACL_SRC_1, _impl->original_b);
            }

            // Release temporary tensors that are only used in prepare stage
            release_temporaries<BITensor>(_impl->aux_mem_req, _impl->workspace);
            _impl->is_prepared = true;
        }
    }
}