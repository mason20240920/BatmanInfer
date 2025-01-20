//
// Created by Mason on 2025/1/20.
//

#include <runtime/experimental/operators/bi_cpu_gemm_lowp.hpp>

#include <data/core/utils/quantization/asymm_helpers.hpp>

#include <cpu/operators/bi_cpu_gemm_lowp_matrix_multiply_core.hpp>

namespace BatmanInfer {
    namespace experimental {
        namespace op {
            struct BICpuGEMMLowp::Impl {
                std::unique_ptr<BatmanInfer::cpu::BICpuGemmLowpMatrixMultiplyCore> op{nullptr};
                bool is_prepared{false};
            };

            BICpuGEMMLowp::BICpuGEMMLowp() : _impl(std::make_unique<Impl>()) {
                _impl->op = std::make_unique<cpu::BICpuGemmLowpMatrixMultiplyCore>();
            }

            BICpuGEMMLowp::~BICpuGEMMLowp() = default;

            experimental::BIMemoryRequirements BICpuGEMMLowp::workspace() const {
                return _impl->op->workspace();
            }

            void BICpuGEMMLowp::configure(
                    const BIITensorInfo *a, const BIITensorInfo *b, const BIITensorInfo *c, BIITensorInfo *output,
                    const GEMMInfo &gemm_info) {
                BI_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

                // Make the B matrix dynamic values.
                auto b_info_to_use = b->clone();
                if (!gemm_info.reshape_b_only_on_first_run()) {
                    b_info_to_use->set_are_values_constant(false);
                }

                _impl->is_prepared = false;
                _impl->op->configure(a, b_info_to_use.get(), (c != nullptr ? c : nullptr), output, gemm_info);
            }

            BIStatus BICpuGEMMLowp::validate(const BIITensorInfo *a,
                                             const BIITensorInfo *b,
                                             const BIITensorInfo *c,
                                             const BIITensorInfo *output,
                                             const GEMMInfo &gemm_info) {
                // Make the B matrix dynamic values.
                auto b_info_to_use = b->clone();
                if (!gemm_info.reshape_b_only_on_first_run()) {
                    b_info_to_use->set_are_values_constant(false);
                }

                return cpu::BICpuGemmLowpMatrixMultiplyCore::validate(a, b_info_to_use.get(), c, output, gemm_info);
            }

            void BICpuGEMMLowp::run(BIITensorPack &tensors) {
                prepare(tensors);
                _impl->op->run(tensors);
            }

            void BICpuGEMMLowp::prepare(BIITensorPack &tensors) {
                if (!_impl->is_prepared) {
                    _impl->op->prepare(tensors);

                    auto aux_mem_req = _impl->op->workspace();

                    auto has_reshape =
                            std::find_if(aux_mem_req.begin(), aux_mem_req.end(),
                                         [](const BIMemoryInfo &m) -> bool {
                                             return m.lifetime == MemoryLifetime::Persistent;
                                         });

                    if (has_reshape != std::end(aux_mem_req)) {
                        auto b = tensors.get_tensor(BITensorType::ACL_SRC_1);
                        b->mark_as_unused();
                    }

                    _impl->is_prepared = true;
                }
            }
        } // namespace op
    }
}