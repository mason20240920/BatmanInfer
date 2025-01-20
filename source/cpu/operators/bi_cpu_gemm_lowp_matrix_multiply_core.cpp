//
// Created by Mason on 2025/1/20.
//

#include <cpu/operators/bi_cpu_gemm_lowp_matrix_multiply_core.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/kernel_descriptors.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/kernels/bi_cpu_convert_quantized_signedness_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_inter_leave_4x4_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_matrix_multiply_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_matrix_reduction_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_offset_contribution_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_lowp_offset_contribution_output_stage_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_transpose_1xw_kernel.hpp>
#include <cpu/operators/bi_cpu_activation.hpp>
#include <cpu/operators/internal/cpu_gemm_assembly_dispatch.hpp>
#include <cpu/utils/bi_cpu_aux_tensor_handler.hpp>

using namespace BatmanInfer::misc::shape_calculator;
using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    namespace cpu {
        namespace {
            cpu::BIAsmGemmInfo init_assembly_metadata(const GEMMInfo &info) {
                cpu::BIAsmGemmInfo asm_info;
                asm_info.method = cpu::BIAsmConvMethod::Im2Col;
                asm_info.reinterpret_input_as_3d = info.reinterpret_input_as_3d();
                asm_info.depth_output_gemm3d = info.depth_output_gemm3d();
                asm_info.activation_info = info.activation_info();
                asm_info.output_stage = info.gemmlowp_output_stage();
                asm_info.fast_mode = info.fast_math();
                asm_info.accumulate = info.accumulate();

                return asm_info;
            }
        } // namespace

        BICpuGemmLowpMatrixMultiplyCore::BICpuGemmLowpMatrixMultiplyCore()
                : _asm_glue(std::make_unique<BICpuGemmAssemblyDispatch>()),
                  _mm_kernel(),
                  _mtx_a_reshape_kernel(),
                  _mtx_b_reshape_kernel(),
                  _mtx_a_reduction_kernel(),
                  _mtx_b_reduction_kernel(),
                  _offset_contribution_kernel(),
                  _offset_contribution_output_stage_kernel(),
                  _activation_func(),
                  _convert_to_signed_asymm(),
                  _convert_from_signed_asymm(),
                  _vector_sum_col(),
                  _vector_sum_row(),
                  _tmp_a(),
                  _tmp_b(),
                  _mm_result_s32(),
                  _signed_a(),
                  _signed_output(),
                  _a_offset(0),
                  _b_offset(0),
                  _run_vector_matrix_multiplication(false),
                  _assembly_path(false),
                  _fused_assembly_path(false),
                  _reshape_b_only_on_first_run(false),
                  _is_prepared(false),
                  _fuse_output_stage(false),
                  _run_activation(false),
                  _flip_signedness(false),
                  _gemm_info(),
                  _aux_mem(Count) {

        }

        BICpuGemmLowpMatrixMultiplyCore::~BICpuGemmLowpMatrixMultiplyCore() = default;

        void BICpuGemmLowpMatrixMultiplyCore::configure(const BatmanInfer::BIITensorInfo *a,
                                                        const BatmanInfer::BIITensorInfo *b,
                                                        const BatmanInfer::BIITensorInfo *c,
                                                        BatmanInfer::BIITensorInfo *dst,
                                                        const BatmanInfer::GEMMInfo &gemm_info) {
            BI_COMPUTE_ERROR_ON_NULLPTR(a, b, dst);
            BI_COMPUTE_ERROR_THROW_ON(BICpuGemmLowpMatrixMultiplyCore::validate(a, b, c, dst, gemm_info));
            BI_COMPUTE_LOG_PARAMS(a, b, c, dst, gemm_info);

            const BIITensorInfo *matrix_a = a;
            const BIITensorInfo *matrix_b = b;
            GEMMInfo info = gemm_info;

            // Set internal variables
            _a_offset = a->quantization_info().uniform().offset;
            _b_offset = b->quantization_info().uniform().offset;
            _run_vector_matrix_multiplication = a->dimension(1) < 2;
            _reshape_b_only_on_first_run = b->are_values_constant();
            _is_prepared = false;
            _fused_assembly_path = false;
            _flip_signedness =
                    is_data_type_quantized_per_channel(b->data_type()) && (a->data_type() == BIDataType::QASYMM8) &&
                    _reshape_b_only_on_first_run;
            _gemm_info = gemm_info;


            const BIITensorInfo *a_to_use = a;

            // Initialize assembly kernel meta-data
            const cpu::BIAsmGemmInfo asm_info = init_assembly_metadata(gemm_info);

            const int32_t offset_correction = 128;
            const BIDataType dt = BIDataType::QASYMM8_SIGNED;
            const BIUniformQuantizationInfo iqinfo = a_to_use->quantization_info().uniform();

            _signed_a = a_to_use->clone()->set_data_type(dt).set_quantization_info(
                    BIQuantizationInfo(iqinfo.scale, iqinfo.offset + offset_correction));

            // If inputs are mixed-sign but this machine does not support mixed sign kernels,
            // flip the sign so matched-sign kernels can be used.
            if (!_flip_signedness && a->data_type() == BIDataType::QASYMM8 &&
                b->data_type() == BIDataType::QASYMM8_SIGNED &&
                !bool(BICpuGemmAssemblyDispatch::validate(a_to_use, b, c, dst, asm_info)
                )) {
                _flip_signedness = true;
            }

            _asm_glue = std::make_unique<cpu::BICpuGemmAssemblyDispatch>();

            // Convert to QASYMM8 -> QASYMM8_SIGNED and back
            if (_flip_signedness) {
                _convert_to_signed_asymm = std::make_unique<kernels::BICpuConvertQuantizedSignednessKernel>();
                _convert_to_signed_asymm->configure(a_to_use, &_signed_a);
                a_to_use = &_signed_a;
                _a_offset = _signed_a.quantization_info().uniform().offset;

                const BIUniformQuantizationInfo oqinfo = dst->quantization_info().uniform();
                _signed_output = dst->clone()->set_data_type(dt).set_quantization_info(
                        BIQuantizationInfo(oqinfo.scale, oqinfo.offset - offset_correction));

                // Output stage correction
                BIGEMMLowpOutputStageInfo output_stage_corr = info.gemmlowp_output_stage();
                output_stage_corr.gemmlowp_offset = _signed_output.quantization_info().uniform().offset;
                output_stage_corr.gemmlowp_min_bound -= offset_correction;
                output_stage_corr.gemmlowp_max_bound -= offset_correction;
                info.set_gemmlowp_output_stage(output_stage_corr);

                // Update matrix a
                matrix_a = &_signed_a;
            }

            // Offset kernel is need if offset is non-zero or it may change (i.e. dynamic).
            // It is not needed if the datatype is symmetric, because there is no offset
            bool a_offset_kernel_needed = _a_offset != 0 || a->quantization_info().is_dynamic();
            bool b_offset_kernel_needed = _b_offset != 0 || b->quantization_info().is_dynamic();

            // If GEMMLowpOutputStage != NONE, fuse the offset contribution with the output stage
            if (info.gemmlowp_output_stage().type != BIGEMMLowpOutputStageType::NONE) {
                _fuse_output_stage = true;
                _mm_result_s32 = BITensorInfo(dst->tensor_shape(), 1, BIDataType::S32);
            }

#ifdef __aarch64__
            if (!(!b->are_values_constant() &&
                  b->tensor_shape().z() > 1)) // Disable batch matmul as optimized GeMM handles batching differently.
            {
                switch (a->data_type()) {
                    case BIDataType::QASYMM8:
                    case BIDataType::QASYMM8_SIGNED:
                    case BIDataType::U8:
                    case BIDataType::S8: {
                        if (is_data_type_quantized_asymmetric(a_to_use->data_type()) &&
                            info.gemmlowp_output_stage().type == BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT) {
                            auto c_info_to_use = c == nullptr ? nullptr : c;
                            _asm_glue->configure(a_to_use, b, c_info_to_use, dst, asm_info);
                            _fused_assembly_path = _asm_glue->is_configured();
                        } else {
                            auto output_to_use = (_fuse_output_stage ? &_mm_result_s32 : dst);
                            _asm_glue->
                                    configure(a_to_use, b,
                                              nullptr, output_to_use, asm_info);
                        }
                        _assembly_path = _asm_glue->is_configured();
                        break;
                    }
                    default: {
                        BI_COMPUTE_ERROR("Datatype not supported");
                        break;
                    }
                }
            }
#endif /* __aarch64__ */
            if (!(_assembly_path || _run_vector_matrix_multiplication)) {
                matrix_a = &_tmp_a;
                matrix_b = &_tmp_b;

                // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
                _tmp_a = BITensorInfo(compute_interleaved_shape(*a_to_use), 1, a_to_use->data_type(),
                                      a_to_use->quantization_info());
                // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
                _tmp_b = BITensorInfo(compute_transpose_1xw_with_element_size_shape(*b), 1, b->data_type(),
                                      b->quantization_info());

                // Configure interleave kernel
                _mtx_a_reshape_kernel = std::make_unique<kernels::BICpuGemmInterleave4x4Kernel>();
                _mtx_a_reshape_kernel->
                        configure(a_to_use, &_tmp_a
                );

                // Configure transpose kernel
                _mtx_b_reshape_kernel = std::make_unique<kernels::BICpuGemmTranspose1xWKernel>();
                _mtx_b_reshape_kernel->configure(b, &_tmp_b);
            }

            if (!_fused_assembly_path) {
                // Build reduction info
                const GEMMLowpReductionKernelInfo reduction_info(a_to_use->dimension(0), false, 0, false);

                if (a_offset_kernel_needed) {
                    _vector_sum_col = BITensorInfo(compute_reductionA_shape(*b), 1, BIDataType::S32);

                    // Configure Matrix B reduction kernel
                    _mtx_b_reduction_kernel = std::make_unique<kernels::BICpuGemmLowpMatrixBReductionKernel>();
                    _mtx_b_reduction_kernel->
                            configure(b, &_vector_sum_col, reduction_info
                    );
                }

                if (b_offset_kernel_needed) {
                    _vector_sum_row = BITensorInfo(compute_reductionB_shape(*a_to_use), 1, BIDataType::S32);

                    // Configure matrix A reduction kernel
                    _mtx_a_reduction_kernel = std::make_unique<kernels::BICpuGemmLowpMatrixAReductionKernel>();
                    _mtx_a_reduction_kernel->
                            configure(a_to_use, &_vector_sum_row, reduction_info
                    );
                }

                if (_fuse_output_stage) {
                    // Configure matrix multiply kernel
                    if (!_assembly_path) {
                        _mm_kernel = std::make_unique<kernels::BICpuGemmLowpMatrixMultiplyKernel>();
                        _mm_kernel->
                                configure(matrix_a, matrix_b, &_mm_result_s32
                        );
                    }

                    _offset_contribution_output_stage_kernel =
                            std::make_unique<kernels::BICpuGemmLowpOffsetContributionOutputStageKernel>();
                    _offset_contribution_output_stage_kernel->configure(&_mm_result_s32,
                                                                        a_offset_kernel_needed ? &_vector_sum_col
                                                                                               : nullptr,
                                                                        b_offset_kernel_needed ? &_vector_sum_row
                                                                                               : nullptr, c,
                                                                        _flip_signedness ? &_signed_output : dst,
                                                                        a->dimension(0), _a_offset, _b_offset,
                                                                        info.gemmlowp_output_stage());

                    if (_flip_signedness) {
                        _convert_from_signed_asymm = std::make_unique<kernels::BICpuConvertQuantizedSignednessKernel>();
                        _convert_from_signed_asymm->
                                configure(&_signed_output, dst
                        );
                    }
                } else {
                    // This scale is needed for the s8_f32 kernel where the multiplication output is dequantized to F32.
                    const float dequantize_scale =
                            (dst->data_type() == BIDataType::F32)
                            ? a->quantization_info().uniform().scale * b->quantization_info().uniform().scale
                            : 1.0f;
                    // Configure matrix multiply kernel
                    if (!_assembly_path) {
                        _mm_kernel = std::make_unique<kernels::BICpuGemmLowpMatrixMultiplyKernel>();
                        _mm_kernel->
                                configure(matrix_a, matrix_b, dst
                        );
                    }
                    // Configure offset contribution kernel
                    _offset_contribution_kernel = std::make_unique<kernels::BICpuGemmLowpOffsetContributionKernel>();
                    _offset_contribution_kernel->
                            configure(dst, a_offset_kernel_needed
                                           ? &_vector_sum_col : nullptr,
                                      b_offset_kernel_needed ? &_vector_sum_row : nullptr,
                                      a_to_use->dimension(0), _a_offset, _b_offset, dequantize_scale);
                }
            }
            // Configure activation
            const BIActivationLayerInfo &activation = gemm_info.activation_info();
            _run_activation =
                    activation.enabled() &&
                    (!_assembly_path || !cpu::BICpuGemmAssemblyDispatch::is_activation_supported(activation));
            if (_run_activation) {
                _activation_func = std::make_unique<BICpuActivation>();
                _activation_func->
                        configure(dst,
                                  nullptr, activation);
            }

            if (_assembly_path) {
                const auto asm_mem_req = _asm_glue->workspace();
                for (
                        unsigned int slot = 0;
                        slot < asm_mem_req.

                                size();

                        ++slot) {
                    _aux_mem[slot] = asm_mem_req[slot];
                }
            }

            // Request memory for LHS and RHS reshape matrix
            _aux_mem[VectorSumCol] = BIMemoryInfo(offset_int_vec(VectorSumCol),
                                                  !_fused_assembly_path && a_offset_kernel_needed
                                                  && _reshape_b_only_on_first_run
                                                  ? MemoryLifetime::Persistent
                                                  : MemoryLifetime::Temporary,
                                                  _vector_sum_col.total_size());
            _aux_mem[VectorSumRow] = BIMemoryInfo(offset_int_vec(VectorSumRow), MemoryLifetime::Temporary,
                                                  _vector_sum_row.total_size());
            _aux_mem[TmpA] = BIMemoryInfo(offset_int_vec(TmpA), MemoryLifetime::Temporary, _tmp_a.total_size());
            _aux_mem[TmpB] = BIMemoryInfo(offset_int_vec(TmpB),
                                          _reshape_b_only_on_first_run ? MemoryLifetime::Persistent
                                                                       : MemoryLifetime::Temporary,
                                          _tmp_b.total_size());
            _aux_mem[MMResultS32] = BIMemoryInfo(offset_int_vec(MMResultS32), MemoryLifetime::Temporary,
                                                 _mm_result_s32.total_size());
            _aux_mem[SignedA] = BIMemoryInfo(offset_int_vec(SignedA), MemoryLifetime::Temporary,
                                             _signed_a.total_size());
            _aux_mem[SignedOutput] = BIMemoryInfo(offset_int_vec(SignedOutput), MemoryLifetime::Temporary,
                                                  _signed_output.total_size());
        }

        BIStatus BICpuGemmLowpMatrixMultiplyCore::validate(const BIITensorInfo *a,
                                                           const BIITensorInfo *b,
                                                           const BIITensorInfo *c,
                                                           const BIITensorInfo *output,
                                                           const GEMMInfo &gemm_info) {
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, BIDataType::QASYMM8, BIDataType::QASYMM8_SIGNED);
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(b, 1, BIDataType::QASYMM8, BIDataType::QASYMM8_SIGNED,
                                                                BIDataType::QSYMM8, BIDataType::QSYMM8_PER_CHANNEL);
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, BIDataType::S32, BIDataType::QASYMM8,
                                                                BIDataType::QASYMM8_SIGNED, BIDataType::F32);
            BI_COMPUTE_RETURN_ERROR_ON_MSG(c != nullptr && output->data_type() != BIDataType::F32 &&
                                           gemm_info.gemmlowp_output_stage().type == BIGEMMLowpOutputStageType::NONE,
                                           "Bias addition not supported in NEGEMMLowpMatrixMultiplyCore for output S32");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(
                    (a)->dimension(0) != (b)->dimension(1),
                    "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.pretranspose_A(),
                                           "Matrix A already pretransposed is not supported");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.pretranspose_B(),
                                           "Matrix B already pretransposed is not supported");

            // When using accumulation(in place summation), for now, the only supported DataType for output is S32.
            if (gemm_info.accumulate()) {
#ifdef __arm__
                BI_COMPUTE_RETURN_ERROR_MSG("Accumulation is not supported for armv7");
#endif /* __arm__ */
                BI_COMPUTE_RETURN_ERROR_ON_MSG(
                        gemm_info.gemmlowp_output_stage().type != BIGEMMLowpOutputStageType::NONE,
                        "Accumulation is not supported for output QASYMM8/QASYMM8_SIGNED");
            }

            GEMMInfo info = gemm_info;
            const BIITensorInfo *matrix_a_info = a;
            const BIITensorInfo *matrix_b_info = b;

            const BIITensorInfo *a_to_use = a;

            BITensorInfo tmp_a_info{};
            BITensorInfo tmp_b_info{};
            BITensorInfo mm_result_s32_info{};

            int32_t a_offset = a->quantization_info().uniform().offset;
            int32_t b_offset = b->quantization_info().uniform().offset;

            bool fuse_output_stage = info.gemmlowp_output_stage().type != BIGEMMLowpOutputStageType::NONE;
            if (fuse_output_stage) {
                auto_init_if_empty(mm_result_s32_info,
                                   a->clone()->set_tensor_shape(output->tensor_shape()).set_data_type(BIDataType::S32));
            }

            // Initialize assembly kernel meta-data
            const BIAsmGemmInfo asm_info = init_assembly_metadata(info);

            // Convert QASYMM8->QASYMM8_SIGNED
            const int32_t offset_correction = 128;
            const BIDataType dt = BIDataType::QASYMM8_SIGNED;
            const BIUniformQuantizationInfo iqinfo = a_to_use->quantization_info().uniform();

            BITensorInfo signed_a = a_to_use->clone()->set_data_type(dt).set_quantization_info(
                    BIQuantizationInfo(iqinfo.scale, iqinfo.offset + offset_correction));
            BITensorInfo signed_output{};

            bool flip_signedness = is_data_type_quantized_per_channel(b->data_type()) &&
                                   (a->data_type() == BIDataType::QASYMM8) && info.reshape_b_only_on_first_run();

            // If inputs are mixed-sign but this machine does not support mixed sign kernels,
            // flip the sign so matched-sign kernels can be used.
            if (!flip_signedness && a->data_type() == BIDataType::QASYMM8 &&
                b->data_type() == BIDataType::QASYMM8_SIGNED &&
                !bool(BICpuGemmAssemblyDispatch::validate(a_to_use, b, c, output, asm_info))) {
                flip_signedness = true;
            }

            if (flip_signedness) {
                BI_COMPUTE_RETURN_ON_ERROR(
                        kernels::BICpuConvertQuantizedSignednessKernel::validate(a_to_use, &signed_a));
                a_to_use = &signed_a;
                a_offset = signed_a.quantization_info().uniform().offset;

                const BIUniformQuantizationInfo oqinfo = output->quantization_info().uniform();
                signed_output = output->clone()->set_data_type(dt).set_quantization_info(
                        BIQuantizationInfo(oqinfo.scale, oqinfo.offset - offset_correction));

                // Output stage correction
                BIGEMMLowpOutputStageInfo output_stage_corr = info.gemmlowp_output_stage();
                output_stage_corr.gemmlowp_offset = signed_output.quantization_info().uniform().offset;
                output_stage_corr.gemmlowp_min_bound -= offset_correction;
                output_stage_corr.gemmlowp_max_bound -= offset_correction;
                info.set_gemmlowp_output_stage(output_stage_corr);

                // Update matrix a
                matrix_a_info = &signed_a;
            }

            // Offset kernel is need if offset is non-zero or it may change (i.e. dynamic).
            bool a_offset_kernel_needed = a_offset != 0 || a->quantization_info().is_dynamic();
            bool b_offset_kernel_needed = b_offset != 0 || b->quantization_info().is_dynamic();

            // Check if we need to run the optimized assembly kernel
            bool run_optimised = false;
            bool run_optimised_requantized = false;

            if (!(!b->are_values_constant() &&
                  b->tensor_shape().z() > 1)) // Disable batch matmul as optimized GeMM handles batching differently.
            {
                if (is_data_type_quantized_asymmetric(a_to_use->data_type()) &&
                    info.gemmlowp_output_stage().type == BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT) {
                    run_optimised = bool(BICpuGemmAssemblyDispatch::validate(a_to_use, b, c, output, asm_info));
                    run_optimised_requantized = run_optimised;
                } else {
                    run_optimised = bool(BICpuGemmAssemblyDispatch::validate(
                            a_to_use, b, nullptr, fuse_output_stage ? &mm_result_s32_info : output, asm_info));
                }
            }

            if (run_optimised) {
                BI_COMPUTE_RETURN_ERROR_ON(b->dimension(0) != output->dimension(0));
                if (info.depth_output_gemm3d() != 0) {
                    if (info.reinterpret_input_as_3d()) {
                        BI_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
                        BI_COMPUTE_RETURN_ERROR_ON(a->dimension(2) != output->dimension(2));
                    } else {
                        BI_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1) * output->dimension(2));
                    }
                } else {
                    BI_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
                }
            } else {
                BI_COMPUTE_RETURN_ERROR_ON_MSG(info.reinterpret_input_as_3d(),
                                               "NEGEMM cannot reinterpret the input tensor as 3D");
                BI_COMPUTE_RETURN_ERROR_ON_MSG(info.depth_output_gemm3d() != 0,
                                               "NEGEMM cannot reinterpret the output tensor as 3D");

                const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
                if (!run_vector_matrix_multiplication) {
                    matrix_a_info = &tmp_a_info;
                    matrix_b_info = &tmp_b_info;

                    // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
                    BITensorShape shape_tmp_a = a->tensor_shape();
                    shape_tmp_a.set(0, a->dimension(0) * 4);
                    shape_tmp_a.set(1, std::ceil(a->dimension(1) / 4.f));

                    // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
                    BITensorShape shape_tmp_b = b->tensor_shape();
                    shape_tmp_b.set(0, b->dimension(1) * 16);
                    shape_tmp_b.set(1, std::ceil(b->dimension(0) / 16.f));

                    // Validate interleave kernel
                    auto_init_if_empty(tmp_a_info, a_to_use->clone()->set_tensor_shape(shape_tmp_a));
                    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(shape_tmp_b));

                    BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuGemmInterleave4x4Kernel::validate(a_to_use, &tmp_a_info));
                    BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuGemmTranspose1xWKernel::validate(b, &tmp_b_info));
                }
            }

            if (!run_optimised_requantized) {
                BITensorInfo info_vector_sum_col{};
                BITensorInfo info_vector_sum_row{};

                const GEMMLowpReductionKernelInfo reduction_info(a_to_use->dimension(0), false, 0, false);

                // Validate matrix B reduction kernel only if _a_offset is not equal to 0
                if (a_offset_kernel_needed) {
                    info_vector_sum_col = BITensorInfo(compute_reductionA_shape(*b), 1, BIDataType::S32);

                    // Configure Matrix B reduction kernel
                    BI_COMPUTE_RETURN_ON_ERROR(
                            kernels::BICpuGemmLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col,
                                                                                   reduction_info));
                }

                // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
                if (b_offset_kernel_needed) {
                    info_vector_sum_row = BITensorInfo(compute_reductionB_shape(*a), 1, BIDataType::S32);

                    // Configure matrix A reduction kernel
                    BI_COMPUTE_RETURN_ON_ERROR(
                            kernels::BICpuGemmLowpMatrixAReductionKernel::validate(a_to_use, &info_vector_sum_row,
                                                                                   reduction_info));
                }

                if (fuse_output_stage) {
                    if (!run_optimised) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                info.reinterpret_input_as_3d(),
                                "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the input tensor as 3D");
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                info.depth_output_gemm3d() != 0,
                                "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the output tensor as 3D");

                        BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuGemmLowpMatrixMultiplyKernel::validate(
                                matrix_a_info, matrix_b_info, &mm_result_s32_info));
                    }

                    // Validate offset contribution kernel
                    BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuGemmLowpOffsetContributionOutputStageKernel::validate(
                            &mm_result_s32_info, a_offset_kernel_needed ? &info_vector_sum_col : nullptr,
                            b_offset_kernel_needed ? &info_vector_sum_row : nullptr, c,
                            flip_signedness ? &signed_output : output,
                            a_offset, b_offset, info.gemmlowp_output_stage()));
                } else {
                    if (!run_optimised) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                info.reinterpret_input_as_3d(),
                                "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the input tensor as 3D");
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                info.depth_output_gemm3d() != 0,
                                "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the output tensor as 3D");

                        BI_COMPUTE_RETURN_ON_ERROR(
                                kernels::BICpuGemmLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info,
                                                                                     output));
                    }
                    // Validate offset contribution kernel
                    if (output->data_type() != BIDataType::QASYMM8 &&
                        output->data_type() != BIDataType::QASYMM8_SIGNED) {
                        BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuGemmLowpOffsetContributionKernel::validate(
                                output, a_offset_kernel_needed ? &info_vector_sum_col : nullptr,
                                b_offset_kernel_needed ? &info_vector_sum_row : nullptr, a_offset, b_offset));
                    }
                }
            }

            // Validate activation
            const BIActivationLayerInfo &activation = gemm_info.activation_info();
            if (activation.enabled()) {
                BI_COMPUTE_RETURN_ON_ERROR(BICpuActivation::validate(output, nullptr, activation));
            }

            return BIStatus{};
        }

        void BICpuGemmLowpMatrixMultiplyCore::run(BIITensorPack &tensors) {
            prepare(tensors);

            auto a = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
            auto b = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
            auto c = tensors.get_const_tensor(BITensorType::ACL_SRC_2);
            auto dst = tensors.get_tensor(BITensorType::ACL_DST);
            auto a_to_use = a;
            auto matrix_a = a;
            auto matrix_b = b;

            CpuAuxTensorHandler vector_sum_col(offset_int_vec(VectorSumCol), _vector_sum_col, tensors, false);
            CpuAuxTensorHandler vector_sum_row(offset_int_vec(VectorSumRow), _vector_sum_row, tensors, false);
            CpuAuxTensorHandler tmp_a(offset_int_vec(TmpA), _tmp_a, tensors, false);
            CpuAuxTensorHandler tmp_b(offset_int_vec(TmpB), _tmp_b, tensors, true);
            CpuAuxTensorHandler mm_result_s32(offset_int_vec(MMResultS32), _mm_result_s32, tensors, false);
            CpuAuxTensorHandler signed_a(offset_int_vec(SignedA), _signed_a, tensors, false);
            CpuAuxTensorHandler signed_output(offset_int_vec(SignedOutput), _signed_output, tensors, false);

            const BIQuantizationInfo a_qinfo = a->info()->quantization_info();
            const BIQuantizationInfo b_qinfo = b->info()->quantization_info();

            if (a_qinfo.is_dynamic())
                _a_offset = a_qinfo.uniform().offset;
            if (b_qinfo.is_dynamic())
                _b_offset = b_qinfo.uniform().offset;

            // Convert QASYMM8->QASYMM8_SIGNED
            if (_flip_signedness) {
                BIITensorPack pack = {{BITensorType::ACL_SRC, a},
                                      {BITensorType::ACL_DST, signed_a.get()}};
                BINEScheduler::get().schedule_op(_convert_to_signed_asymm.get(), BIWindow::DimY,
                                                 _convert_to_signed_asymm->window(),
                                                 pack);
                a_to_use = signed_a.get();
                matrix_a = signed_a.get();
            }

            // Run GEMM
            if (_asm_glue->is_configured()) {
                BIITensorPack asm_glue_tensors = tensors;
                if (is_data_type_quantized_asymmetric(a_to_use->info()->data_type()) &&
                    _gemm_info.gemmlowp_output_stage().type == BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT) {
                    asm_glue_tensors.add_const_tensor(BITensorType::ACL_SRC_0, a_to_use);
                    asm_glue_tensors.add_const_tensor(BITensorType::ACL_SRC_1, b);
                    asm_glue_tensors.add_const_tensor(BITensorType::ACL_SRC_2, c);
                    asm_glue_tensors.add_tensor(BITensorType::ACL_DST, dst);
                } else {
                    auto output_to_use = (_fuse_output_stage ? mm_result_s32.get() : dst);
                    asm_glue_tensors.add_const_tensor(BITensorType::ACL_SRC_0, a_to_use);
                    asm_glue_tensors.add_const_tensor(BITensorType::ACL_SRC_1, b);
                    asm_glue_tensors.add_tensor(BITensorType::ACL_DST, output_to_use);
                }
                _asm_glue->run(asm_glue_tensors);
            } else {
                if (!_run_vector_matrix_multiplication) {
                    matrix_a = tmp_a.get();
                    matrix_b = tmp_b.get();
                    // Run interleave kernel
                    BIITensorPack pack_a = {{BITensorType::ACL_SRC, a_to_use},
                                            {BITensorType::ACL_DST, tmp_a.get()}};
                    BINEScheduler::get().schedule_op(_mtx_a_reshape_kernel.get(), BIWindow::DimY,
                                                     _mtx_a_reshape_kernel->window(),
                                                     pack_a);

                    if (!_reshape_b_only_on_first_run) {
                        BIITensorPack pack_b = {{BITensorType::ACL_SRC, b},
                                                {BITensorType::ACL_DST, tmp_b.get()}};
                        // Run transpose kernel
                        BINEScheduler::get().schedule_op(_mtx_b_reshape_kernel.get(), BIWindow::DimY,
                                                         _mtx_b_reshape_kernel->window(), pack_b);
                    }
                }
                BIITensorPack pack_mm = {{BITensorType::ACL_SRC_0, matrix_a},
                                         {BITensorType::ACL_SRC_1, matrix_b}};
                if (_fuse_output_stage) {
                    pack_mm.add_tensor(BITensorType::ACL_DST, mm_result_s32.get());
                } else {
                    pack_mm.add_tensor(BITensorType::ACL_DST, dst);
                }
                BINEScheduler::get().schedule_op(_mm_kernel.get(), BIWindow::DimY, _mm_kernel->window(), pack_mm);
            }

            if (!_fused_assembly_path) {
                // Run matrix A reduction kernel only if _b_offset is not equal to 0
                if (_b_offset != 0) {
                    BIITensorPack pack = {{BITensorType::ACL_SRC, a_to_use},
                                          {BITensorType::ACL_DST, vector_sum_row.get()}};
                    BINEScheduler::get().schedule_op(_mtx_a_reduction_kernel.get(), BIWindow::DimX,
                                                     _mtx_a_reduction_kernel->window(), pack);
                }

                // Run matrix B reduction kernel only if _a_offset is not equal to 0
                if (_a_offset != 0 && !_reshape_b_only_on_first_run) {
                    BIITensorPack pack = {{BITensorType::ACL_SRC, b},
                                          {BITensorType::ACL_DST, vector_sum_col.get()}};
                    BINEScheduler::get().schedule_op(_mtx_b_reduction_kernel.get(), BIWindow::DimX,
                                                     _mtx_b_reduction_kernel->window(), pack);
                }

                if (_fuse_output_stage) {
                    if (a_qinfo.is_dynamic())
                        _offset_contribution_output_stage_kernel->set_a_offset(_a_offset);
                    if (b_qinfo.is_dynamic())
                        _offset_contribution_output_stage_kernel->set_b_offset(_b_offset);

                    BIITensorPack pack;
                    pack.add_tensor(BITensorType::ACL_SRC_0, mm_result_s32.get());
                    pack.add_tensor(BITensorType::ACL_SRC_1, _a_offset == 0 ? nullptr : vector_sum_col.get());
                    pack.add_tensor(BITensorType::ACL_SRC_2, _b_offset == 0 ? nullptr : vector_sum_row.get());
                    pack.add_tensor(BITensorType::ACL_SRC_3, c);
                    pack.add_tensor(BITensorType::ACL_DST, _flip_signedness ? signed_output.get() : dst);

                    // Run offset contribution kernel
                    BINEScheduler::get().schedule_op(_offset_contribution_output_stage_kernel.get(), BIWindow::DimY,
                                                     _offset_contribution_output_stage_kernel->window(), pack);
                } else {
                    if (a_qinfo.is_dynamic())
                        _offset_contribution_kernel->set_a_offset(_a_offset);
                    if (b_qinfo.is_dynamic())
                        _offset_contribution_kernel->set_b_offset(_b_offset);
                    if (a_qinfo.is_dynamic() || b_qinfo.is_dynamic()) {
                        const float dequantize_scale = a_qinfo.uniform().scale * b_qinfo.uniform().scale;
                        _offset_contribution_kernel->set_scale(dequantize_scale);
                    }

                    BIITensorPack pack;
                    pack.add_tensor(BITensorType::ACL_SRC_0, _a_offset == 0 ? nullptr : vector_sum_col.get());
                    pack.add_tensor(BITensorType::ACL_SRC_1, _b_offset == 0 ? nullptr : vector_sum_row.get());
                    pack.add_tensor(BITensorType::ACL_DST, dst);

                    // Run offset contribution kernel
                    BINEScheduler::get().schedule_op(_offset_contribution_kernel.get(), BIWindow::DimY,
                                                     _offset_contribution_kernel->window(), pack);
                }
            }

            // Convert QASYMM8_SIGNED->QASYMM8
            if (!_fused_assembly_path && _fuse_output_stage && _flip_signedness) {
                BIITensorPack pack = {{BITensorType::ACL_SRC, signed_output.get()},
                                      {BITensorType::ACL_DST, dst}};
                BINEScheduler::get().schedule_op(_convert_from_signed_asymm.get(), BIWindow::DimY,
                                                 _convert_from_signed_asymm->window(), pack);
            }

            // Run fused activation unless already run in the fused assembly
            if (_run_activation) {
                BIITensorPack pack = {{BITensorType::ACL_SRC, dst},
                                      {BITensorType::ACL_DST, dst}};
                _activation_func->run(pack);
            }
        }

        void BICpuGemmLowpMatrixMultiplyCore::prepare(BIITensorPack &tensors) {
            if (!_is_prepared) {
                auto original_b = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                // Run assembly reshape
                if (_asm_glue->is_configured()) {
                    _asm_glue->prepare(tensors);
                }
                    // Run non-assembly reshape
                else if (_reshape_b_only_on_first_run && !_run_vector_matrix_multiplication &&
                         !_asm_glue->is_configured()) {
                    // Run reshape kernel and mark original weights tensor as unused
                    BIITensor *tmp_b_p = utils::cast::polymorphic_downcast<BIITensor *>(
                            tensors.get_tensor(offset_int_vec(TmpB)));
                    CpuAuxTensorHandler tmp_b(_tmp_b, *tmp_b_p);
                    BIITensorPack pack = {{BITensorType::ACL_SRC, original_b},
                                          {BITensorType::ACL_DST, tmp_b.get()}};
                    BINEScheduler::get().schedule_op(_mtx_b_reshape_kernel.get(), BIWindow::DimY,
                                                     _mtx_b_reshape_kernel->window(),
                                                     pack);
                }

                // Run matrix B reduction kernel only if _a_offset is not equal to 0
                if (!_fused_assembly_path && _a_offset != 0 && _reshape_b_only_on_first_run) {
                    BIITensor *vector_sum_col_p =
                            utils::cast::polymorphic_downcast<BIITensor *>(
                                    tensors.get_tensor(offset_int_vec(VectorSumCol)));
                    CpuAuxTensorHandler vector_sum_col(_vector_sum_col, *vector_sum_col_p);
                    BIITensorPack pack = {{BITensorType::ACL_SRC, original_b},
                                          {BITensorType::ACL_DST, vector_sum_col.get()}};
                    BINEScheduler::get().schedule_op(_mtx_b_reduction_kernel.get(), BIWindow::DimX,
                                                     _mtx_b_reduction_kernel->window(), pack);
                }
                _is_prepared = true;
            }
        }

        experimental::BIMemoryRequirements BICpuGemmLowpMatrixMultiplyCore::workspace() const {
            return _aux_mem;
        }

        void
        BICpuGemmLowpMatrixMultiplyCore::update_quantization_parameters(const BIGEMMLowpOutputStageInfo &output_info,
                                                                        const BIQuantizationInfo &a,
                                                                        const BIQuantizationInfo &b,
                                                                        const bool is_prepared,
                                                                        const bool negated_offsets) {
            auto lowp_os = output_info;
            _gemm_info.set_gemmlowp_output_stage(lowp_os);
            _asm_glue->update_quantization_parameters(output_info, a, b, is_prepared, negated_offsets);
            _is_prepared = is_prepared;
        }
    } // namespace cpu
}