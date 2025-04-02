//
// Created by Mason on 2025/1/16.
//

#include  <cpu/operators/bi_cpu_mat_mul.hpp>

#include <data/core/experimental/types.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/utils/quantization/asymm_helpers.hpp>
#include <data/core/bi_vlidate.hpp>
#include <function_info/bi_MatMulInfo.h>
#include <runtime/neon/functions/bi_ne_mat_mul.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <data/core/utils/quantization/asymm_helpers.hpp>
#include <cpu/utils/bi_cpu_aux_tensor_handler.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>

using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    namespace cpu {
        namespace {
            /**
             * 计算和配置 GEMMLowp 输出阶段的参数
             * @param src 源张量（输入特征图）的信息
             * @param weights 权重张量的信息
             * @param dst 目标张量（输出特征图）的信息
             * @param act 激活函数信息（如 ReLU 等）
             * @param gemmlowp_output_stage_info 存储输出阶段配置信息的结构体
             * @return
             */
            BIStatus get_gemmlowp_output_stage_info(const BIITensorInfo *src,
                                                    const BIITensorInfo *weights,
                                                    const BIITensorInfo *dst,
                                                    const BIActivationLayerInfo &act,
                                                    BIGEMMLowpOutputStageInfo &gemmlowp_output_stage_info) {
                // 提取输入张量、权重张量和输出张量的量化信息
                // 量化信息包括 scale 和 offset，分别表示量化的比例因子和零点
                // data_type 表示张量的数据类型（如 int8、uint8）
                const auto data_type = src->data_type();
                const BIQuantizationInfo oq_info = dst->quantization_info();
                const BIUniformQuantizationInfo iq_unif = src->quantization_info().uniform();
                const BIUniformQuantizationInfo wq_unif = weights->quantization_info().uniform();
                const BIUniformQuantizationInfo oq_unif = oq_info.uniform();

                // multiplier 是一个浮点数，表示输入张量和权重张量的量化比例因子与输出张量的比例因子的关系。
                // 它用于将低精度计算结果重新映射到输出张量的量化范围
                float multiplier = (iq_unif.scale * wq_unif.scale) / oq_unif.scale;
                int32_t output_multiplier;
                int32_t output_shift;

                BI_COMPUTE_RETURN_ON_ERROR(
                        quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));

                int32_t type_min = 0;
                int32_t type_max = 0;
                std::tie(type_min, type_max) = quantization::get_quantized_asymmetric_output_min_max(oq_info, act,
                                                                                                     data_type);

                // 量化乘法器
                gemmlowp_output_stage_info.gemmlowp_multiplier = output_multiplier;
                // 移位值
                gemmlowp_output_stage_info.gemmlowp_shift = output_shift;
                // 输出张量的零点，用于调整结果的基准值
                gemmlowp_output_stage_info.gemmlowp_offset = oq_unif.offset;
                gemmlowp_output_stage_info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
                gemmlowp_output_stage_info.gemmlowp_min_bound = type_min;
                gemmlowp_output_stage_info.gemmlowp_max_bound = type_max;

                return BIStatus{};
            }
        } // namespace

        BICpuMatMul::BICpuMatMul()
                : _transpose_kernel_lhs(),
                  _transpose_kernel_rhs(),
                  _asm_glue(),
                  _lhs_transposed(),
                  _rhs_transposed(),
                  _original_lhs_shape(),
                  _original_rhs_shape(),
                  _original_dst_shape() {
        }

        BIStatus BICpuMatMul::validate(const BatmanInfer::BIITensorInfo *lhs, const BatmanInfer::BIITensorInfo *rhs,
                                       const BatmanInfer::BIITensorInfo *dst, const BatmanInfer::BIMatMulInfo &info,
                                       const BatmanInfer::BICpuMatMulSettings &settings,
                                       const BatmanInfer::BIActivationLayerInfo &act_info) {
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs, dst);
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, BIDataType::F32, BIDataType::F16,
                                                                BIDataType::BFLOAT16,
                                                                BIDataType::QASYMM8, BIDataType::QASYMM8_SIGNED);
            BI_COMPUTE_RETURN_ERROR_ON_MSG(lhs->are_values_constant(), "LHS Tensor must be dynamic.");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(rhs->are_values_constant(), "RHS Tensor must be dynamic.");
            BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(lhs);
            BI_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(lhs);

            const auto adj_lhs = info.adj_lhs();
            const auto adj_rhs = info.adj_rhs();

            const BIITensorInfo *lhs_to_use = lhs;
            const BIITensorInfo *rhs_to_use = rhs;
            BITensorInfo lhs_transposed{};
            BITensorInfo rhs_transposed{};

            auto gemm_info = BIAsmGemmInfo();
            gemm_info.activation_info = act_info;
            gemm_info.fast_mode = settings.fast_math();
            gemm_info.fixed_format = settings.fixed_format();

            // Validate and then permute a/b
            if (adj_lhs) {
                auto_init_if_empty(lhs_transposed,
                                   lhs->clone()->set_tensor_shape(
                                           misc::shape_calculator::compute_transposed_shape(*lhs)));
                BI_COMPUTE_RETURN_ON_ERROR(cpu::kernels::BICpuTransposeKernel::validate(lhs_to_use, &lhs_transposed));
                // Assign lhs_to_use pointer to use transposed TensorInfo
                lhs_to_use = &lhs_transposed;
            }

            if (adj_rhs) {
                auto_init_if_empty(rhs_transposed,
                                   rhs->clone()->set_tensor_shape(
                                           misc::shape_calculator::compute_transposed_shape(*rhs)));
                BI_COMPUTE_RETURN_ON_ERROR(cpu::kernels::BICpuTransposeKernel::validate(rhs_to_use, &rhs_transposed));
                // Assign rhs_to_use pointer to use transposed TensorInfo
                rhs_to_use = &rhs_transposed;
            }

            auto ret = lhs_to_use->dimension(0);
            auto new_ret = rhs_to_use->dimension(1);

            BI_COMPUTE_RETURN_ERROR_ON_MSG(lhs_to_use->dimension(0) != rhs_to_use->dimension(1),
                                           "The product AB is defined only if the number of columns in A is equal to the "
                                           "number of rows in B (after transpose)");


            // Iterate over dimensions to be collapsed in operator - check dimensions are equivalent between tensors
            for (unsigned int i = 2; i < BICoordinates::num_max_dimensions; i++) {
                BI_COMPUTE_RETURN_ERROR_ON_MSG(lhs_to_use->dimension(i) != rhs_to_use->dimension(i),
                                               "Broadcasting in Batch dimension is unsupported by this operator.");
            }

            // Quantized-specific configuration
            if (is_data_type_quantized(lhs->data_type())) {
                BI_COMPUTE_RETURN_ON_ERROR(get_gemmlowp_output_stage_info(lhs_to_use, rhs_to_use, dst,
                                                                          gemm_info.activation_info,
                                                                          gemm_info.output_stage));
            }

            if (gemm_info.fixed_format) {
                gemm_info.weight_format = BIWeightFormat::ANY;
                BatmanInfer::BIWeightFormat expected_weight_format = BIWeightFormat::ANY;
                BI_COMPUTE_RETURN_ON_ERROR(
                        cpu::BICpuGemmAssemblyDispatch::has_opt_impl(expected_weight_format, lhs_to_use,
                                                                     rhs_to_use, nullptr, dst, gemm_info));

                // Set gemm weights info to the one returned by has_opt_impl because the user query the kernel for the format to be set.
                gemm_info.weight_format = expected_weight_format;
            }

            BI_COMPUTE_RETURN_ON_ERROR(
                    cpu::BICpuGemmAssemblyDispatch::validate(lhs_to_use, rhs_to_use, nullptr, dst, gemm_info));

            return BIStatus{};
        }

        void BICpuMatMul::configure(BIITensorInfo *lhs,
                                    BIITensorInfo *rhs,
                                    BIITensorInfo *dst,
                                    const BIMatMulInfo &info,
                                    const BICpuMatMulSettings &settings,
                                    const BIActivationLayerInfo &act_info) {
            BI_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
            BI_COMPUTE_LOG_PARAMS(lhs, rhs, dst, info, settings);
            BI_COMPUTE_ERROR_THROW_ON(BICpuMatMul::validate(lhs, rhs, dst, info, settings)); // 先验证能不能使用MatMul

            _adj_lhs = info.adj_lhs();
            _adj_rhs = info.adj_rhs();
            _fast_math = settings.fast_math();

            // 1. 创建和重置张量
            // ------------------------------------------------------
            // a. 初始化阶段拷贝TensorInfo信息，防止修改输入张量被修改
            // b. 修改lhs/dst的数据格式到[x, y, 1, collapsed(z)]适配汇编核 kernel configuration
            // c. For rhs collapse all dimensions larger than 3 to z dimension
            BITensorInfo lhs_to_use = *lhs->clone();
            BITensorInfo dst_to_use = *dst->clone();
            BITensorInfo rhs_to_use = *rhs->clone();

            // 保存初始时候的张量形状信息
            _original_lhs_shape = lhs_to_use.tensor_shape();
            _original_dst_shape = dst_to_use.tensor_shape();
            _original_rhs_shape = rhs_to_use.tensor_shape();

            // Reshape lhs for use with assembly kernels.
            lhs_to_use.set_tensor_shape(
                    BITensorShape(_original_lhs_shape.x(), _original_lhs_shape.y(), 1,
                                  _original_lhs_shape.collapsed_from(2).z()));
            dst_to_use.set_tensor_shape(
                    BITensorShape(_original_dst_shape.x(), _original_dst_shape.y(), 1,
                                  _original_dst_shape.collapsed_from(2).z()));
            rhs_to_use.set_tensor_shape(_original_rhs_shape.collapsed_from(2));

            // 2.  配置AB矩阵的转置
            // ------------------------------------------------------
            // Initialise transposed TensorInfo class for aux tensors (intermediary tensors)
            if (_adj_lhs) {
                // Setup transpose LHS
                _transpose_kernel_lhs = std::make_unique<cpu::kernels::BICpuTransposeKernel>();
                _transpose_kernel_lhs->configure(&lhs_to_use, &_lhs_transposed);

                _aux_mem[TransposeLHS] = BIMemoryInfo(offset_int_vec(TransposeLHS), MemoryLifetime::Temporary,
                                                      lhs->total_size());
            }

            if (_adj_rhs) {
                // Setup transpose RHS
                _transpose_kernel_rhs = std::make_unique<cpu::kernels::BICpuTransposeKernel>();
                _transpose_kernel_rhs->configure(&rhs_to_use, &_rhs_transposed);

                _aux_mem[TransposeRHS] = BIMemoryInfo(offset_int_vec(TransposeRHS), MemoryLifetime::Temporary,
                                                      rhs->total_size());
            }

            // 3. Configure assembly kernel using transposed tensors.
            // -----------------------------------------------------
            // Use transposed tensors if the corresponding transpose flags are set
            // Fill AsmGemmInfo class object before configuration
            _gemm_info.activation_info = act_info;
            _gemm_info.fast_mode = settings.fast_math();
            _gemm_info.fixed_format = settings.fixed_format();
            _gemm_info.negated_offsets = false;

            lhs_to_use = (_adj_lhs) ? _lhs_transposed : lhs_to_use;
            rhs_to_use = (_adj_rhs) ? _rhs_transposed : rhs_to_use;

            // Quantized-specific configuration
            if (is_data_type_quantized(lhs->data_type())) {
                get_gemmlowp_output_stage_info(&lhs_to_use, &rhs_to_use, &dst_to_use, _gemm_info.activation_info,
                                               _gemm_info.output_stage);
            }

            if (_gemm_info.fixed_format) {
                _gemm_info.weight_format = BIWeightFormat::ANY;
                BatmanInfer::BIWeightFormat expected_weight_format = BIWeightFormat::ANY;
                BIStatus ret = cpu::BICpuGemmAssemblyDispatch::has_opt_impl(expected_weight_format, &lhs_to_use,
                                                                            &rhs_to_use,
                                                                            nullptr, dst, _gemm_info);
                BI_COMPUTE_ERROR_THROW_ON(ret);

                // Set gemm weights info to the one returned by has_opt_impl because the user query the kernel for the format to be set.
                _gemm_info.weight_format = expected_weight_format;
                // has_opt_impl may return a non fast math kernel, even if we requested one
                _gemm_info.fast_mode = BatmanInfer::is_fixed_format_fast_math(expected_weight_format);
            }

            // 配置汇编核
            _asm_glue = std::make_unique<cpu::BICpuGemmAssemblyDispatch>();
            _asm_glue->configure(&lhs_to_use, &rhs_to_use, nullptr, &dst_to_use,
                                 _gemm_info); // c is nullptr as bias not supported in MatMul

            BI_COMPUTE_EXIT_ON_MSG(!_asm_glue->is_configured(), "Error in CpuGemmAssemblyDispatch configuration");
            // Specify memory requirements for intermediate tensors
            auto asm_mem_req = _asm_glue->workspace();
            // Specify memory required by gemm kernel
            int idx = 0;
            // 1. 现在默认只会第一次创建修改B矩阵的转置
            // 2. 所以现在对B矩阵进行动态创建
            // 3. B矩阵的逻辑是aux.slot = 1026
            for (const auto &aux: asm_mem_req) {
                _aux_mem[idx] = aux;
                idx++;
            }
        }

        void BICpuMatMul::run(BIITensorPack &tensors) {
            // Retrieve tensors from tensor pack
            auto lhs = tensors.get_tensor(ACL_SRC_0);
            auto rhs = tensors.get_const_tensor(ACL_SRC_1);
            auto dst = tensors.get_tensor(ACL_DST);

            // Update for dynamic tensors (dynamic batch size and dynamic sequence length
            BITensorInfo lhs_to_use = *lhs->info()->clone();
            BITensorInfo dst_to_use = *dst->info()->clone();
            BITensorInfo rhs_to_use = *rhs->info()->clone();

            // 保存初始时候的张量形状信息
            _original_lhs_shape = lhs_to_use.tensor_shape();
            _original_dst_shape = dst_to_use.tensor_shape();
            _original_rhs_shape = rhs_to_use.tensor_shape();

            // Reshape lhs for use with assembly kernels.
            lhs_to_use.set_tensor_shape(
                    BITensorShape(_original_lhs_shape.x(), _original_lhs_shape.y(), 1,
                                  _original_lhs_shape.collapsed_from(2).z()));
            dst_to_use.set_tensor_shape(
                    BITensorShape(_original_dst_shape.x(), _original_dst_shape.y(), 1,
                                  _original_dst_shape.collapsed_from(2).z()));
            rhs_to_use.set_tensor_shape(_original_rhs_shape.collapsed_from(2));

            // 获取动态的aux结果(但是他会在prepare阶段初始化放在tensors里面
            size_t tensor_b_align;
            auto tensor_size = _asm_glue->dynamic_tensor_b_size(&lhs_to_use, &rhs_to_use, &dst_to_use, tensor_b_align);
            // 算子的动态的数据格式
            auto _tmp_data_type = lhs->info()->data_type();
            
            tensors.remove_tensor(1026);
            BITensor transpose_b_tensor;
            transpose_b_tensor.allocator()->init(BITensorInfo(BITensorShape(tensor_size), 1, _tmp_data_type),
                                                 tensor_b_align);
            transpose_b_tensor.allocator()->allocate();
            tensors.add_tensor(1026, &transpose_b_tensor);


            // 配置汇编核
//            _asm_glue = std::make_unique<cpu::BICpuGemmAssemblyDispatch>();
//            _asm_glue->configure(&lhs_to_use, &rhs_to_use, nullptr, &dst_to_use,
//                                 _gemm_info); // c is nullptr as bias not supported in MatMul
//
//            BI_COMPUTE_EXIT_ON_MSG(!_asm_glue->is_configured(), "Error in CpuGemmAssemblyDispatch configuration");

            // Reshape LHS and DST to ensure compatibility with GEMM asm kernel (Batch dimensions is 4th for lhs and dst within asm)
            // Collapse RHS (necessary to support dimensions larger than 3 in gemm assembly)
            lhs->info()->set_tensor_shape(
                    BITensorShape(_original_lhs_shape.x(), _original_lhs_shape.y(), 1,
                                  _original_lhs_shape.collapsed_from(2).z())); // Collapsed 3+ dimensions into z
            dst->info()->set_tensor_shape(
                    BITensorShape(_original_dst_shape.x(), _original_dst_shape.y(), 1,
                                  _original_dst_shape.collapsed_from(2).z())); // Collapsed 3+ dimensions into z
            rhs->info()->set_tensor_shape(_original_rhs_shape.collapsed_from(2));

            // Initialise object to handle stored transposed tensors in auxillary memory
            CpuAuxTensorHandler lhs_transposed(offset_int_vec(TransposeLHS), _lhs_transposed, tensors, true);
            CpuAuxTensorHandler rhs_transposed(offset_int_vec(TransposeRHS), _rhs_transposed, tensors, true);

            // Create tensor pack for asm kernel
            BIITensorPack asm_tensors(tensors);

            // Run transpose lhs if necessary
            if (_adj_lhs) {
                BIITensorPack lhs_transpose_pack = {{BITensorType::ACL_SRC, lhs},
                                                    {BITensorType::ACL_DST, lhs_transposed.get()}};
                BINEScheduler::get().schedule_op(_transpose_kernel_lhs.get(), BIWindow::DimY,
                                                 _transpose_kernel_lhs->window(),
                                                 lhs_transpose_pack);
                asm_tensors.add_const_tensor(BITensorType::ACL_SRC_0, lhs_transposed.get());
            }
            // Run transpose rhs if necessary
            if (_adj_rhs) {
                BIITensorPack rhs_transpose_pack = {{BITensorType::ACL_SRC, rhs},
                                                    {BITensorType::ACL_DST, rhs_transposed.get()}};
                BINEScheduler::get().schedule_op(_transpose_kernel_rhs.get(), BIWindow::DimY,
                                                 _transpose_kernel_rhs->window(),
                                                 rhs_transpose_pack);
                asm_tensors.add_const_tensor(BITensorType::ACL_SRC_1, rhs_transposed.get());
            }
            // Run asm kernel
            _asm_glue->run(asm_tensors);

            // Undo reshape of tensors
            dst->info()->set_tensor_shape(_original_dst_shape);
            lhs->info()->set_tensor_shape(_original_lhs_shape);
            rhs->info()->set_tensor_shape(_original_rhs_shape);
        }

        experimental::BIMemoryRequirements BICpuMatMul::workspace() const {
            return _aux_mem;
        }

        const experimental::BIMemoryRequirements &BICpuMatMul::workspace_dynamic(const BIITensorPack &tensors) const {
            BI_COMPUTE_ERROR_ON(tensors.empty());
            BI_COMPUTE_ERROR_ON(_adj_lhs || _adj_rhs); // 如果需要转置操作就进行报错
            // Update memory requirements with those from the kernel.
            _aux_mem.reserve(Count);
            _aux_mem.resize(Count);

//            for (BIMemoryInfo mi: _kernel->workspace(tensors)) {
//                _aux_mem.push_back(mi);
//            }

            return _aux_mem;
        }


    } // namespace cpu
} // namespace BatmanInfer