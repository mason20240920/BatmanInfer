//
// Created by Mason on 2025/3/27.
//

#include <cpu/kernels/bi_cpu_dynamic_gemm_kernel.hpp>

#include <data/core/bi_vlidate.hpp>
#include <function_info/bi_GEMMInfo.h>

#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/kernels/dynamic_gemm/heuristics/bi_cpu_dynamic_gemm_kernel_heuristics.hpp>

using namespace BatmanInfer::experimental;
using namespace BatmanInfer::cpu::kernels::heuristics;

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {

            void BICpuDynamicGemmKernel::configure(const BIITensorInfo *a,
                                                   const BIITensorInfo *b,
                                                   const BIITensorInfo *c,
                                                   BIITensorInfo *d,
                                                   float alpha,
                                                   float beta,
                                                   size_t base_aux_slot,
                                                   const GEMMInfo &gemm_info) {
                BI_COMPUTE_ERROR_THROW_ON(BICpuDynamicGemmKernel::validate(a, b, c, d, alpha, beta, gemm_info));

                _heuristics = BICpuDynamicGemmKernelHeuristics{a, b, c, d, alpha, beta, gemm_info};

                _name = std::string{"BICpuDynamicGemmKernel"}.append("/").append(_heuristics.name());

                _base_aux_slot = base_aux_slot;
                _aux_mem.reserve(Count);

                BIWindow window = _heuristics.get_window()(d);
                BIICPPKernel::configure(window);
            }

            BIStatus BICpuDynamicGemmKernel::validate(const BIITensorInfo *a,
                                                      const BIITensorInfo *b,
                                                      const BIITensorInfo *c,
                                                      const BIITensorInfo *d,
                                                      float alpha,
                                                      float beta,
                                                      const GEMMInfo &gemm_info) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(a, b, c, d);
                BI_COMPUTE_UNUSED(d);
                BI_COMPUTE_UNUSED(alpha);
                BI_COMPUTE_UNUSED(beta);
                BI_COMPUTE_UNUSED(gemm_info);

                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, BIDataType::F32);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, c, d);

                // If both a and b are static, so are c and d, rendering this kernel moot.
                BI_COMPUTE_RETURN_ERROR_ON(!a->is_dynamic() && !b->is_dynamic());
                // ...conversely, when either a or b is dynamic, so is d.
                BI_COMPUTE_RETURN_ERROR_ON(!d->is_dynamic());
                // What remains that could possibly be static is exactly one of a or b, and
                // optionally c. Dimensions are checked in run_op.

                // We expect to be able to pre-pack b and c if the values are constant, so
                // they must be static.
                if (b->are_values_constant()) {
                    BI_COMPUTE_RETURN_ERROR_ON(b->is_dynamic());
                }
                if (c->are_values_constant()) {
                    BI_COMPUTE_RETURN_ERROR_ON(c->is_dynamic());
                }

                BI_COMPUTE_RETURN_ERROR_ON(alpha != 1.0f);
                BI_COMPUTE_RETURN_ERROR_ON(beta != 1.0f);

                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.is_a_reshaped());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.is_b_reshaped());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.reshape_b_only_on_first_run() &&
                                           (!b->are_values_constant() || !c->are_values_constant()));
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.depth_output_gemm3d() != 0);
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.reinterpret_input_as_3d());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.retain_internal_weights());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.gemmlowp_output_stage() != BIGEMMLowpOutputStageInfo{});
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.fast_math());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.fp_mixed_precision());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.broadcast_bias());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.pretranspose_A());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.pretranspose_B());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.activation_info() != BIActivationLayerInfo{});
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.fixed_format());
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.weight_format() != BIWeightFormat::UNSPECIFIED);
                BI_COMPUTE_RETURN_ERROR_ON(gemm_info.accumulate());

                return BIStatus{};
            }

            void
            BICpuDynamicGemmKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_EXIT_ON_MSG(tensors.empty(), "No inputs provided");

                BIICPPKernel::configure(window);

                const BIITensor *a = tensors.get_const_tensor(ACL_SRC_0);
                const BIITensor *b = tensors.get_const_tensor(ACL_SRC_1);
                const BIITensor *c = tensors.get_const_tensor(ACL_SRC_2);
                BIITensor *d = tensors.get_tensor(ACL_DST);
                BIITensor *pack_b = tensors.get_tensor(offset_int_vec(_base_aux_slot + PackedRHS));

                BI_COMPUTE_EXIT_ON_MSG(
                        a->info()->dimension(0) != b->info()->dimension(1),
                        "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
                BI_COMPUTE_EXIT_ON_MSG(a->info()->dimension(1) != d->info()->dimension(1),
                                       "The number of rows in Output must equal the number of rows in Lhs");
                BI_COMPUTE_EXIT_ON_MSG(b->info()->dimension(0) != d->info()->dimension(0),
                                       "The number of columns in Output must equal the number of columns in Rhs");
                BI_COMPUTE_EXIT_ON_MSG(c->info()->dimension(0) != d->info()->dimension(0),
                                       "The number of columns in Output must equal the number of columns in Bias");
                BI_COMPUTE_EXIT_ON_MSG(c->info()->dimension(1) != 1, "Bias must be a vector");

                _heuristics.kernel()(a, b, c, d, pack_b, window);
            }

            const char *BICpuDynamicGemmKernel::name() const {
                return _name.c_str();
            }

            const BIMemoryRequirements &BICpuDynamicGemmKernel::workspace(const BIITensorPack &tensors) const {
                BI_COMPUTE_ERROR_ON(tensors.empty());

                const BIITensor *const b = tensors.get_const_tensor(ACL_SRC_1);
                BI_COMPUTE_ERROR_ON_NULLPTR(b);

                // The ukernel needs a tensor allocation for the packed RHS.
                const BITensorShape &b_shape = b->info()->tensor_shape();
                const size_t pack_b_size = _heuristics.size_of_packed_rhs()(b_shape.y(), b_shape.x());
                _aux_mem[PackedRHS] = BIMemoryInfo{offset_int_vec(_base_aux_slot + PackedRHS),
                                                   MemoryLifetime::Persistent,
                                                   std::max(pack_b_size, size_t{1})};

                return _aux_mem;
            }

            void BICpuDynamicGemmKernel::prepare(BIITensorPack &tensors, const bool reuse_b) {
                const BIITensor *const dst = tensors.get_const_tensor(ACL_DST);
                BIWindow window = _heuristics.get_window()(dst->info());
                BIICPPKernel::configure(window);

                const bool run_packing = !reuse_b;
                if (run_packing) {
                    const BIITensor *const rhs = tensors.get_const_tensor(ACL_SRC_1);
                    const BIITensor *const bias = tensors.get_const_tensor(ACL_SRC_2);
                    const int pack_b_tensor_offset = offset_int_vec(_base_aux_slot + PackedRHS);
                    BIITensor *const pack_b = tensors.get_tensor(pack_b_tensor_offset);

                    _heuristics.pack_rhs()(rhs, bias, pack_b);
                }
            }

            size_t BICpuDynamicGemmKernel::size_of_packed_rhs(size_t rows, size_t columns) const {
                // TODO: finished in future
                return 0;
            }

        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer