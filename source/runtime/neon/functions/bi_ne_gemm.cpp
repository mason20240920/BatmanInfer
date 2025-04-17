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
#include <cpu/operators/bi_cpu_dynamic_gemm.hpp>

using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    namespace {
        /**
         * @brief 根据tensor info信息判断是不是动态算子
         * 如果包含-1就是动态维度
         * @param a
         * @param b
         * @param c
         * @param d
         * @return
         */
        inline bool is_dynamic(
            const BIITensorInfo *a,
            const BIITensorInfo *b,
            const BIITensorInfo *c,
            const BIITensorInfo *d
        ) {
            if (!c) {
                return a->is_dynamic() || b->is_dynamic() || d->is_dynamic();
            }
            return a->is_dynamic() || b->is_dynamic() || c->is_dynamic() || d->is_dynamic();
        }

        /**
         * @brief 根据tensor来查看是不是动态算子
         * @param a
         * @param b
         * @param c
         * @param d
         * @return
         */
        inline bool is_dynamic(
            const BIITensor *a,
            const BIITensor *b,
            const BIITensor *c,
            const BIITensor *d) {
            if (!c) {
                return is_dynamic(a->info(), b->info(), nullptr, d->info());
            }
            return is_dynamic(a->info(), b->info(), c->info(), d->info());
        }

        std::unique_ptr<cpu::BIICpuOperator> make_and_config_op(const BIITensorInfo *a,
                                                                const BIITensorInfo *b,
                                                                const BIITensorInfo *c,
                                                                BIITensorInfo *d,
                                                                float alpha,
                                                                float beta,
                                                                const GEMMInfo &gemm_info) {
            // 让B矩阵拥有动态值
            auto b_info_to_use = b->clone();
            if (!gemm_info.reshape_b_only_on_first_run())
                b_info_to_use->set_are_values_constant(false);

            std::unique_ptr<cpu::BIICpuOperator> op;
            if (is_dynamic(a, b, c, d)) {
                auto op_typed = std::make_unique<cpu::BICpuDynamicGemm>();
                op_typed->configure(a, b_info_to_use.get(), c, d, alpha, beta, gemm_info);
                op = std::move(op_typed);
            } else {
                auto op_typed = std::make_unique<cpu::BICpuGemm>();
                op_typed->configure(a, b_info_to_use.get(), c, d, alpha, beta, gemm_info);
                op = std::move(op_typed);
            }
            return op;
        }
    }

    struct BINEGEMM::Impl {
        BIMemoryGroup memory_group{};
        BIIWeightsManager *weights_manager{nullptr};

        std::unique_ptr<cpu::BICpuGemm> op{nullptr};

        const BIITensor *original_b{nullptr};
        bool is_prepared{false};
        bool is_dynamic{false};

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

        _impl->is_dynamic = is_dynamic(a, b, c, d); // 确定是不是动态张量

        BI_COMPUTE_ERROR_THROW_ON(
            cpu::BICpuGemm::validate(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr,
                d->info(), alpha, beta, gemm_info));

        // Check if we need to reshape the matrix B only on the first run
        _impl->is_prepared = false;
        // _impl->memory_group.mappings().clear();
        _impl->original_b = b;
        // 让B矩阵拥有动态值
        auto b_info_to_use = b->info()->clone();
        if (!gemm_info.reshape_b_only_on_first_run())
            b_info_to_use->set_are_values_constant(false);

        auto op_typed = std::make_unique<cpu::BICpuGemm>();
        op_typed->configure(a->info(), b_info_to_use.get(), c != nullptr ? c->info() : nullptr, d->info(), alpha, beta,
                            gemm_info);
        _impl->op = std::move(op_typed);

        // _impl->op = make_and_config_op(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr, d->info(), alpha,
        //                                beta, gemm_info);

        _impl->run_pack = {
            {ACL_SRC_0, a},
            {ACL_SRC_1, b},
            {ACL_SRC_2, c},
            {ACL_DST, d}
        };
        _impl->prep_pack = {
            {ACL_SRC_1, b},
            {ACL_SRC_2, c}
        };

        if (_impl->is_dynamic)
            // 第一次获取并不太关注张量的大小, 因为它们是在 run() 中重新分配的，
            // 而是关于哪些张量将会在工作区管理。
            _impl->aux_mem_req = _impl->op->workspace_dynamic(_impl->run_pack);
        else
            _impl->aux_mem_req = _impl->op->workspace();
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
    BINEGEMM::has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                           const BatmanInfer::BIITensorInfo *a,
                           const BatmanInfer::BIITensorInfo *b, const BatmanInfer::BIITensorInfo *c,
                           const BatmanInfer::BIITensorInfo *output, float alpha, float beta,
                           const BatmanInfer::GEMMInfo &gemm_info) {
        BI_COMPUTE_UNUSED(alpha, beta);
        BI_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(a, b, c, output);

        return cpu::BICpuGemm::has_opt_impl(expected_weight_format, a, b, c, output, gemm_info);
    }

    void BINEGEMM::dynamic_configure() {
        auto a = _impl->run_pack.get_const_tensor(ACL_SRC_0);
        auto b = _impl->run_pack.get_const_tensor(ACL_SRC_1);
        auto d = _impl->run_pack.get_tensor(ACL_DST);
        _impl->op->dynamic_configure(a->info(), b->info(), d->info());
    }


    void BINEGEMM::run() {
        prepare();

        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        _impl->op->run(_impl->run_pack);
    }

    void BINEGEMM::prepare() {
        // 如果是动态, 就进行动态内存管理
        if (_impl->is_dynamic) {
            _impl->aux_mem_req = _impl->op->workspace_dynamic(_impl->run_pack);
            reallocate_tensors(_impl->aux_mem_req, _impl->workspace); // 动态分配内存
        } else if (!_impl->is_prepared) {
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
