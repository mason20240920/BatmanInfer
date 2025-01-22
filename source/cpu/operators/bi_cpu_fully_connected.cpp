//
// Created by Mason on 2025/1/22.
//

#include <cpu/operators/bi_cpu_fully_connected.hpp>

#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/utils/quantization/asymm_helpers.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <data/core/utils/quantization/asymm_helpers.hpp>
#include <cpu/kernels/bi_cpu_transpose_kernel.hpp>
#include <cpu/operators/bi_cpu_convert_fully_connected_weights.hpp>
#include <cpu/operators/bi_cpu_flatten.hpp>
#include <cpu/operators/bi_cpu_gemm.hpp>
#include <cpu/operators/bi_cpu_gemm_lowp_matrix_multiply_core.hpp>
#include <cpu/utils/bi_cpu_aux_tensor_handler.hpp>

namespace BatmanInfer {
    namespace cpu {
        using namespace BatmanInfer::experimental;
        using namespace BatmanInfer::misc::shape_calculator;

        namespace {
            BIStatus get_gemmlowp_output_stage_info(const BIITensorInfo *src,
                                                    const BIITensorInfo *weights,
                                                    const BIITensorInfo *dst,
                                                    const BIActivationLayerInfo &act,
                                                    BIGEMMLowpOutputStageInfo &gemmlowp_output_stage_info) {
                const auto data_type = src->data_type();
                const BIQuantizationInfo oq_info = dst->quantization_info();
                const BIUniformQuantizationInfo iq_unif = src->quantization_info().uniform();
                const BIUniformQuantizationInfo wq_unif = weights->quantization_info().uniform();
                const BIUniformQuantizationInfo oq_unif = oq_info.uniform();

                float multiplier = (iq_unif.scale * wq_unif.scale) / oq_unif.scale;
                int32_t output_multiplier;
                int32_t output_shift;

                BI_COMPUTE_RETURN_ON_ERROR(
                        quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));

                int32_t type_min = 0;
                int32_t type_max = 0;
                std::tie(type_min, type_max) = quantization::get_quantized_asymmetric_output_min_max(oq_info, act,
                                                                                                     data_type);

                gemmlowp_output_stage_info.gemmlowp_multiplier = output_multiplier;
                gemmlowp_output_stage_info.gemmlowp_shift = output_shift;
                gemmlowp_output_stage_info.gemmlowp_offset = oq_unif.offset;
                gemmlowp_output_stage_info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
                gemmlowp_output_stage_info.gemmlowp_min_bound = type_min;
                gemmlowp_output_stage_info.gemmlowp_max_bound = type_max;

                return BIStatus{};
            }

            BIStatus validate_mm(const BIITensorInfo *src,
                                 const BIITensorInfo *weights,
                                 const BIITensorInfo *biases,
                                 const BIITensorInfo *dst,
                                 const BIActivationLayerInfo &act,
                                 bool enable_fast_math,
                                 BIWeightFormat weight_format) {
                if (is_data_type_quantized_asymmetric(src->data_type())) {
                    // 由于计算卷积需要负偏移量，我们需要更改 QuantizationInfo()
                    // 提取并取反 src 和 weights 偏移量
                    const BIQuantizationInfo src_quantization_info(src->quantization_info().uniform().scale,
                                                                   -src->quantization_info().uniform().offset);
                    const BIQuantizationInfo weights_quantization_info(weights->quantization_info().uniform().scale,
                                                                       -weights->quantization_info().uniform().offset);

                    BIGEMMLowpOutputStageInfo gemmlowp_output_stage_info;
                    BI_COMPUTE_RETURN_ON_ERROR(
                            get_gemmlowp_output_stage_info(src, weights, dst, act, gemmlowp_output_stage_info));

                    GEMMInfo gemm_info;
                    gemm_info.set_gemmlowp_output_stage(gemmlowp_output_stage_info);
                    gemm_info.set_fast_math(enable_fast_math);

                    // Validate gemmlowp function
                    BITensorInfo src_info = src->clone()->set_quantization_info(src_quantization_info);
                    BITensorInfo weights_info = weights->clone()->set_quantization_info(weights_quantization_info);
                    BI_COMPUTE_RETURN_ON_ERROR(
                            BICpuGemmLowpMatrixMultiplyCore::validate(&src_info, &weights_info, biases, dst,
                                                                      gemm_info));
                } else {
                    GEMMInfo gemm_info;
                    gemm_info.set_weight_format(weight_format);
                    gemm_info.set_fixed_format(weight_format != BIWeightFormat::UNSPECIFIED);
                    gemm_info.set_fast_math(enable_fast_math);
                    BI_COMPUTE_RETURN_ON_ERROR(BICpuGemm::validate(src, weights, biases, dst, 1.f, 1.0f, gemm_info));
                }

                return BIStatus{};
            }
        } // namespace

        BICpuFullyConnected::BICpuFullyConnected() : _flatten(nullptr),
                                                     _convert_weights(nullptr),
                                                     _transpose_weights(nullptr),
                                                     _mm_gemm(nullptr),
                                                     _mm_gemmlowp(nullptr),
                                                     _flattened_src(),
                                                     _converted_weights(),
                                                     _reshaped_weights(),
                                                     _trans_weights(),
                                                     _trans_weights_idx(AuxTensorIdx::Count),
                                                     _aux_mem(Count),
                                                     _needs_weights_conversion(false),
                                                     _needs_weights_reshape(false),
                                                     _is_fc_after_conv(false),
                                                     _is_quantized_asymmetric(false),
                                                     _is_prepared(false),
                                                     _enable_fast_math(false),
                                                     _fixed_format(false),
                                                     _weight_format(BatmanInfer::BIWeightFormat::UNSPECIFIED),
                                                     _dynamic_weights(false) {
        }

        BICpuFullyConnected::~BICpuFullyConnected() = default;

        void BICpuFullyConnected::configure_mm(const BatmanInfer::BIITensorInfo *src,
                                               const BatmanInfer::BIITensorInfo *weights,
                                               const BatmanInfer::BIITensorInfo *biases,
                                               BatmanInfer::BIITensorInfo *dst,
                                               const BatmanInfer::BIActivationLayerInfo &act) {
            if (_is_quantized_asymmetric) {
                // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
                // Extract and negate src and weights offset
                const BIQuantizationInfo src_quantization_info(src->quantization_info().uniform().scale,
                                                               -src->quantization_info().uniform().offset);
                const BIQuantizationInfo weights_quantization_info(weights->quantization_info().uniform().scale,
                                                                   -weights->quantization_info().uniform().offset);

                BITensorInfo src_info = src->clone()->set_quantization_info(src_quantization_info);
                BITensorInfo weights_info = weights->clone()->set_quantization_info(weights_quantization_info);

                // Configure gemmlowp function and output stage for asymmetric quantized types
                BIGEMMLowpOutputStageInfo gemmlowp_output_stage_info;
                const BIStatus status =
                        get_gemmlowp_output_stage_info(&src_info, &weights_info, dst, act, gemmlowp_output_stage_info);
                BI_COMPUTE_ERROR_ON(status.error_code() != BIErrorCode::OK);

                GEMMInfo gemm_info;
                gemm_info.set_gemmlowp_output_stage(gemmlowp_output_stage_info);
                gemm_info.set_activation_info(act);
                gemm_info.set_fast_math(_enable_fast_math);
                _mm_gemmlowp = std::make_unique<BICpuGemmLowpMatrixMultiplyCore>();
                _mm_gemmlowp->configure(&src_info, &weights_info, biases, dst, gemm_info);
            } else {
                // Configure matrix multiply kernel
                GEMMInfo gemm_info;
                gemm_info.set_activation_info(act);
                gemm_info.set_fast_math(_enable_fast_math);
                gemm_info.set_fixed_format(_fixed_format);
                gemm_info.set_weight_format(_weight_format);
                _mm_gemm = std::make_unique<BICpuGemm>();
                _mm_gemm->configure(src, weights, biases, dst, 1.f, 1.0f, gemm_info);
            }
        }

        void BICpuFullyConnected::configure_conv_fc(const BIITensorInfo *src,
                                                    const BIITensorInfo *weights,
                                                    const BIITensorInfo *biases,
                                                    BIITensorInfo *dst,
                                                    const BIActivationLayerInfo &act) {
            BI_COMPUTE_ERROR_ON(
                    (weights->dimension(1) != (src->dimension(0) * src->dimension(1) * src->dimension(2))));

            // If the fully connected layer is called after a convolution layer, the src tensor must be linearized

            // Initialize output tensor for flatten
            auto_init_if_empty(_flattened_src, src->clone()->set_tensor_shape(compute_flatten_shape(src)));

            _flatten = std::make_unique<BICpuFlatten>();
            _flatten->configure(src, &_flattened_src);

            // Configure matrix multiply kernel
            configure_mm(&_flattened_src, weights, biases, dst, act);
        }

        void BICpuFullyConnected::configure_fc_fc(const BIITensorInfo *src,
                                                  const BIITensorInfo *weights,
                                                  const BIITensorInfo *biases,
                                                  BIITensorInfo *dst,
                                                  const BIActivationLayerInfo &act) {
            BI_COMPUTE_ERROR_ON(src->dimension(0) != weights->dimension(1));

            // Configure matrix multiply kernel
            configure_mm(src, weights, biases, dst, act);
        }

        void BICpuFullyConnected::configure(const BIITensorInfo *src,
                                            const BIITensorInfo *weights,
                                            const BIITensorInfo *biases,
                                            BIITensorInfo *dst,
                                            BIFullyConnectedLayerInfo fc_info,
                                            const BIWeightsInfo &weights_info) {
            // Perform validate step
            BI_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
            BI_COMPUTE_ERROR_THROW_ON(
                    BICpuFullyConnected::validate(src, weights, biases != nullptr ? biases : nullptr, dst, fc_info,
                                                  weights_info));
            BI_COMPUTE_LOG_PARAMS(src, weights, biases, dst, fc_info);

            _needs_weights_conversion = false;
            _needs_weights_reshape = fc_info.transpose_weights && !fc_info.are_weights_reshaped;
            _needs_weights_reshape = _needs_weights_reshape && !fc_info.retain_internal_weights;
            _is_fc_after_conv = true;
            _is_quantized_asymmetric = is_data_type_quantized_asymmetric(src->data_type());
            _is_prepared = false;
            _trans_weights_idx = AuxTensorIdx::Count;
            _enable_fast_math = fc_info.enable_fast_math;
            _fixed_format = weights_info.weight_format() != BIWeightFormat::UNSPECIFIED;
            _weight_format = weights_info.weight_format();
            _dynamic_weights = !weights->are_values_constant() && _needs_weights_reshape;

            // With the Fully Connected layer we can have 4 different cases:
            //  1) Convolution layer -> Fully Connected layer without batches
            //  2) Fully Connected layer -> Fully Connected layer without batches
            //  3) Convolution layer -> Fully Connected layer with batches
            //  4) Fully Connected layer -> Fully Connected layer with batches

            const BIITensorInfo *weights_to_use = weights;

            // Check if we have a fully connected layer with batches
            const bool is_batched_fc_layer = dst->dimension(1) > 1;
            if (is_batched_fc_layer) {
                _is_fc_after_conv = (std::equal(src->tensor_shape().cbegin() + 3, src->tensor_shape().cend(),
                                                dst->tensor_shape().cbegin() + 1));
            } else {
                _is_fc_after_conv = src->num_dimensions() > 1;
            }

            // Reshape weights if needed
            if (_needs_weights_reshape) {
                // Reshape the weights
                _transpose_weights = std::make_unique<kernels::BICpuTransposeKernel>();
                _transpose_weights->configure(weights, &_reshaped_weights);
                _reshaped_weights.set_are_values_constant(weights->are_values_constant());

                weights_to_use = &_reshaped_weights;
                _trans_weights_idx = AuxTensorIdx::TransposedWeights;
            }

            // Convert weights if needed
            if (_is_fc_after_conv && (src->data_layout() != fc_info.weights_trained_layout)) {
                // Convert weights
                _convert_weights = std::make_unique<BICpuConvertFullyConnectedWeights>();
                _convert_weights->configure(weights_to_use, &_converted_weights, src->tensor_shape(),
                                            fc_info.weights_trained_layout);
                _converted_weights.set_are_values_constant(weights_to_use->are_values_constant());

                weights_to_use = &_converted_weights;
                _needs_weights_conversion = true;
                _trans_weights_idx = AuxTensorIdx::ConvertedWeights;
            }

            if (_is_fc_after_conv) {
                // Fully Connected layer after a Convolution Layer without batches
                configure_conv_fc(src, weights_to_use, biases, dst, fc_info.activation_info);
            } else {
                // Fully Connected layer after a Fully Connected Layer without batches
                configure_fc_fc(src, weights_to_use, biases, dst, fc_info.activation_info);
            }

            // Retain the tensorinfo with the weights to use
            if (_needs_weights_reshape || _needs_weights_conversion) {
                _trans_weights = *weights_to_use;
            }

            // Set auxiliary memory requirements
            auto gemm_mem_req = (_is_quantized_asymmetric) ? _mm_gemmlowp->workspace() : _mm_gemm->workspace();
            for (unsigned int i = 0; i < gemm_mem_req.size(); ++i) {
                _aux_mem[i] = gemm_mem_req[i];
            }

            if (_aux_mem[Pretranspose].size > 0) {
                // Release permuted weights at the end of prepare as they are further transposed by the assembly dispatch
                // Do not release them if biases are dynamic and data type is quantized, since the weights tensor will be used for biases offset calculation
                // Keep all the auxiliary tensors in case of dynamic weights as they are recalculated every time.
                _aux_mem[TransposedWeights] = BIMemoryInfo(
                        offset_int_vec(TransposedWeights),
                        _dynamic_weights ? MemoryLifetime::Temporary
                                         : (_is_quantized_asymmetric && biases && !(biases->are_values_constant()))
                                           ? MemoryLifetime::Persistent
                                           : MemoryLifetime::Prepare,
                        _reshaped_weights.total_size());

                _aux_mem[ConvertedWeights] = BIMemoryInfo(offset_int_vec(ConvertedWeights),
                                                          _dynamic_weights ? MemoryLifetime::Temporary
                                                                           : MemoryLifetime::Prepare,
                                                          _converted_weights.total_size());
            } else {
                _aux_mem[TransposedWeights] = BIMemoryInfo(offset_int_vec(TransposedWeights),
                                                           _dynamic_weights ? MemoryLifetime::Temporary
                                                                            : _needs_weights_conversion
                                                                              ? MemoryLifetime::Prepare
                                                                              : MemoryLifetime::Persistent,
                                                           _reshaped_weights.total_size());

                _aux_mem[ConvertedWeights] = BIMemoryInfo(
                        offset_int_vec(ConvertedWeights),
                        _dynamic_weights ? MemoryLifetime::Temporary : MemoryLifetime::Persistent,
                        _converted_weights.total_size());
            }
            _aux_mem[FlattenedSrc] =
                    BIMemoryInfo(offset_int_vec(FlattenedSrc), MemoryLifetime::Temporary, _flattened_src.total_size());
        }

        BIStatus BICpuFullyConnected::has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                                                   const BIITensorInfo *src,
                                                   const BIITensorInfo *weights,
                                                   const BIITensorInfo *biases,
                                                   const BIITensorInfo *dst,
                                                   const BIFullyConnectedLayerInfo &fc_info,
                                                   BIWeightsInfo weights_info) {
            GEMMInfo gemm_info;
            gemm_info.set_activation_info(fc_info.activation_info);
            gemm_info.set_fast_math(fc_info.enable_fast_math);
            gemm_info.set_fixed_format(weights_info.weight_format() != BIWeightFormat::UNSPECIFIED);
            gemm_info.set_weight_format(weights_info.weight_format());

            return BICpuGemm::has_opt_impl(expected_weight_format, src, weights, biases, dst, gemm_info);
        }

        BIStatus BICpuFullyConnected::validate(const BIITensorInfo *src,
                                               const BIITensorInfo *weights,
                                               const BIITensorInfo *biases,
                                               const BIITensorInfo *dst,
                                               const BIFullyConnectedLayerInfo &fc_info,
                                               const BIWeightsInfo &weights_info) {
            BI_COMPUTE_UNUSED(fc_info.retain_internal_weights);
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8, BIDataType::QASYMM8_SIGNED,
                                                                BIDataType::F16, BIDataType::F32);

            if (is_fixed_format_fast_math(weights_info.weight_format())) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(src, BIDataType::F32);
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(weights, BIDataType::BFLOAT16);
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(dst, BIDataType::F32);
            } else {
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights, dst);
            }

            BI_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);
            BI_COMPUTE_RETURN_ERROR_ON(
                    fc_info.activation_info.enabled() && is_data_type_quantized(src->data_type()) &&
                    fc_info.activation_info.activation() != BIActivationLayerInfo::ActivationFunction::RELU &&
                    fc_info.activation_info.activation() != BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU &&
                    fc_info.activation_info.activation() != BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);

            bool weights_reshaped = !fc_info.transpose_weights || fc_info.are_weights_reshaped;
            bool is_fc_after_conv = true;

            const BIITensorInfo &flatten_src =
                    BITensorInfo(src->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(
                            compute_flatten_shape(src)));
            const BIITensorInfo &reshaped_weights = BITensorInfo(
                    weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(
                            compute_transposed_shape(*weights)));
            const BIITensorInfo &converted_weights = weights_reshaped
                                                     ? BITensorInfo(
                            weights->clone()->set_is_resizable(true).reset_padding())
                                                     : BITensorInfo(*reshaped_weights.clone());

            // With the Fully Connected layer we can have 4 different cases:
            //  1) Convolution layer -> Fully Connected layer without batches
            //  2) Fully Connected layer -> Fully Connected layer without batches
            //  3) Convolution layer -> Fully Connected layer with batches
            //  4) Fully Connected layer -> Fully Connected layer with batches

            const BIITensorInfo *src_to_use = src;
            const BIITensorInfo *weights_to_use = weights;

            // Check if we have a fully connected layer with batches
            const bool is_batched_fc_layer = dst->dimension(1) > 1;

            if (biases != nullptr) {
                BI_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
                if (is_data_type_quantized(src->data_type())) {
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, BIDataType::S32);
                } else {
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
                }
            }

            if (is_batched_fc_layer) {
                is_fc_after_conv = (std::equal(src->tensor_shape().cbegin() + 3, src->tensor_shape().cend(),
                                               dst->tensor_shape().cbegin() + 1));
            } else {
                is_fc_after_conv = src->num_dimensions() > 1;
            }

            if (!weights_reshaped) {
                // Validate reshape weights kernel
                BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuTransposeKernel::validate(weights, &reshaped_weights));
                weights_to_use = &reshaped_weights;
            }

            if (is_fc_after_conv && (src->data_layout() != fc_info.weights_trained_layout)) {
                // Validate convert weights kernel
                BI_COMPUTE_RETURN_ON_ERROR(BICpuConvertFullyConnectedWeights::validate(
                        weights_to_use, &converted_weights, src->tensor_shape(), fc_info.weights_trained_layout));
                weights_to_use = &converted_weights;
            }

            if (is_fc_after_conv) {
                // Fully Connected layer after a Convolution Layer without batches
                BI_COMPUTE_RETURN_ERROR_ON(
                        (weights_to_use->dimension(1) != (src->dimension(0) * src->dimension(1) * src->dimension(2))));

                // Validate flatten kernel
                BI_COMPUTE_RETURN_ON_ERROR(BICpuFlatten::validate(src, &flatten_src));
                src_to_use = &flatten_src;
            } else {
                // Fully Connected layer after a Fully Connected Layer without batches
                BI_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != weights_to_use->dimension(1));
            }
            // Validate matrix multiply kernel
            BI_COMPUTE_RETURN_ON_ERROR(validate_mm(src_to_use, weights_to_use, biases, dst, fc_info.activation_info,
                                                   fc_info.enable_fast_math, weights_info.weight_format()));

            return BIStatus{};
        }

        void BICpuFullyConnected::run(BatmanInfer::BIITensorPack &tensors) {
            // 首先调用 prepare 函数，确保所有必要的准备工作已经完成
            prepare(tensors);

#ifdef BI_COMPUTE_ASSERTS_ENABLED
            // 调试模式下记录 run 调用次数
            ++_asrt_run_count;
            // 如果权重是动态的（_dynamic_weights == true），则断言 prepare 和 run 的调用次数必须相等
            BI_COMPUTE_ERROR_ON(_dynamic_weights && _asrt_prepare_count != _asrt_run_count);
#endif // BI_COMPUTE_ASSERTS_ENABLED

            // 从张量包中获取输入源张量，标识符为 ACL_SRC_0
            auto src = tensors.get_const_tensor(ACL_SRC_0);

            // 创建辅助张量处理器，用于管理展平后的输入张量（flattened_src）和可能的权重量化或转置结果（transformed_wei）
            CpuAuxTensorHandler flattened_src(offset_int_vec(FlattenedSrc), _flattened_src, tensors, false);
            CpuAuxTensorHandler transformed_wei(offset_int_vec(_trans_weights_idx), _trans_weights, tensors, false);

            // 如果当前全连接层紧跟在卷积层之后（_is_fc_after_conv 为 true）
            if (_is_fc_after_conv) {
                // 准备展平操作的数据包：源张量为 src，目标张量为 flattened_src
                BIITensorPack flatten_pack{{ACL_SRC, src},
                                           {ACL_DST, flattened_src.get()}};
                // 运行展平操作，将卷积输出的多维张量展平为一维张量
                _flatten->run(flatten_pack);
            }

            // 创建新的张量包 gemm_pack，用于矩阵乘法操作
            BIITensorPack gemm_pack = tensors;
            // 如果展平操作已执行，则将展平后的张量作为输入，否则直接使用原始输入张量
            gemm_pack.add_const_tensor(ACL_SRC_0, (_is_fc_after_conv) ? flattened_src.get() : src);

            // 如果需要对权重进行转置或数据类型转换，则添加转换后的权重张量到 gemm_pack
            if (_needs_weights_reshape || _needs_weights_conversion) {
                gemm_pack.add_const_tensor(ACL_SRC_1, transformed_wei.get());
            }

            // 根据量化类型选择运行低精度或普通矩阵乘法内核
            if (_is_quantized_asymmetric)
                // 如果是非对称量化，运行低精度矩阵乘法内核
                _mm_gemmlowp->run(gemm_pack);
            else
                // 否则运行普通矩阵乘法内核
                _mm_gemm->run(gemm_pack);

        }

        void BICpuFullyConnected::prepare(BatmanInfer::BIITensorPack &tensors) {
            // 检查是否需要准备操作: 如果尚未准备好或权重是动态的，则进行准备
            if (!_is_prepared || _dynamic_weights) {
#ifdef  BI_COMPUTE_ASSERTS_ENABLED
                // 调试模式下记录 prepare 调用次数
                ++_asrt_prepare_count;
                // 如果权重是静态的，但prepare调用多次，则出发报错
                BI_COMPUTE_ERROR_ON(!_dynamic_weights && _asrt_prepare_count > 1);
#endif

                // 从张量包中获取权重张量，标识符维 ACL_SRC_1
                auto weights = tensors.get_const_tensor(ACL_SRC_1);

                // 创建辅助张量处理器，用于管理转置后的权重张量
                CpuAuxTensorHandler reshaped_weights(offset_int_vec(TransposedWeights), _reshaped_weights, tensors,
                                                     false);
                CpuAuxTensorHandler converted_weights(offset_int_vec(ConvertedWeights), _converted_weights, tensors,
                                                      false);

                // 初始化当前权重指针. 初始时指向原始权重张量
                const BIITensor *cur_weights = weights;

                // 如果需要对权重进行转置 (_needs_weights_reshape 为 true)
                if (_needs_weights_reshape) {
                    // 准备转置操作的数据包：源张量为 weights，目标张量为 reshaped_weights
                    BIITensorPack transpose_pack{{ACL_SRC, weights},
                                                 {ACL_DST, reshaped_weights.get()}};
                    // 调用转置内核，使用调度器在 Window::DimY 维度上完成转置
                    BINEScheduler::get().schedule_op(_transpose_weights.get(), BIWindow::DimY,
                                                     _transpose_weights->window(),
                                                     transpose_pack);

                    // 标记原始权重张量为未使用，释放其资源
                    cur_weights->mark_as_unused();
                    // 更新当前权重指针为转置后的权重张量
                    cur_weights = reshaped_weights.get();
                }

                // 如果需要对权重进行数据类型转换（_needs_weights_conversion 为 true）
                if (_needs_weights_conversion) {
                    // 准备转换操作的数据包：源张量为 cur_weights，目标张量为 converted_weights
                    BIITensorPack convert_pack{{ACL_SRC, cur_weights},
                                               {ACL_DST, converted_weights.get()}};
                    // 调用转换内核，完成权重数据类型的转换
                    _convert_weights->run(convert_pack);

                    // 标记当前权重张量为未使用，释放其资源
                    cur_weights->mark_as_unused();
                    // 更新当前权重指针为转换后的权重张量
                    cur_weights = converted_weights.get();
                }

                // 创建一个新的张量包 gemm_pack，将当前权重张量 cur_weights 添加到包中
                BIITensorPack gemm_pack = tensors;
                gemm_pack.add_const_tensor(ACL_SRC_1, cur_weights);

                // 根据是否为非对称量化，选择不同的矩阵乘法内核进行准备
                if (!_is_quantized_asymmetric) {
                    // 普通矩阵乘法内核的准备
                    _mm_gemm->prepare(gemm_pack);
                } else {
                    // 低精度（非对称量化）矩阵乘法内核的准备
                    _mm_gemmlowp->prepare(gemm_pack);
                }

                // 标记为已准备，避免重复执行准备阶段
                _is_prepared = true;
            }
        }

        experimental::BIMemoryRequirements BICpuFullyConnected::workspace() const {
            return _aux_mem;
        }
    } // namespace cpu
}