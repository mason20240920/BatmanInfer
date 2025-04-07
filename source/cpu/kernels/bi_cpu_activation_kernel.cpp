//
// Created by Mason on 2025/1/11.
//

#include "cpu/kernels/bi_cpu_activation_kernel.hpp"

#include "data/core/bi_i_tensor.hpp"

#include "data/core/cpp/bi_cpp_validate.hpp"
#include "data/core/helpers/bi_auto_configuration.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"
#include "cpu/kernels/activation/heuristics/bi_cpu_activation_kernel_heuristics.hpp"
#include "cpu/kernels/activation/list.hpp"

#include <array>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                /* Supported activation in the 8-bit integer domain */
                static const std::array<BIActivationLayerInfo::ActivationFunction, 8> qasymm8_activations = {
                    BIActivationLayerInfo::ActivationFunction::RELU,
                    BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                    BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                    BIActivationLayerInfo::ActivationFunction::LOGISTIC,
                    BIActivationLayerInfo::ActivationFunction::TANH,
                    BIActivationLayerInfo::ActivationFunction::HARD_SWISH,
                    BIActivationLayerInfo::ActivationFunction::LEAKY_RELU,
                    BIActivationLayerInfo::ActivationFunction::GELU,
                };

                /* Static quantization can only, currently, support relu based activations */
                static const std::array<BIActivationLayerInfo::ActivationFunction, 3> qasymm8_static_quant_activations =
                {
                    BIActivationLayerInfo::ActivationFunction::RELU,
                    BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                    BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                };

                /* Supported activation in the 16-bit integer domain */
                static const std::array<BIActivationLayerInfo::ActivationFunction, 4> qsymm16_activations = {
                    BIActivationLayerInfo::ActivationFunction::LOGISTIC,
                    BIActivationLayerInfo::ActivationFunction::TANH,
                    BIActivationLayerInfo::ActivationFunction::HARD_SWISH,
                    BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                };

                BIStatus validate_arguments(const BIITensorInfo *src,
                                            const BIITensorInfo *dst,
                                            const BIActivationLayerInfo &activation_info) {
                    BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::QASYMM8,
                                                                        BIDataType::QSYMM16, BIDataType::F16,
                                                                        BIDataType::F32);

                    heuristics::BICpuActivationKernelHeuristics heuristics(src, dst, activation_info);
                    const auto *uk = heuristics.kernel();
                    BI_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

                    const BIDataType data_type = src->data_type();
                    const BIQuantizationInfo &oq_info = (dst != nullptr)
                                                            ? dst->quantization_info()
                                                            : src->quantization_info();
                    const BIActivationLayerInfo::ActivationFunction f_act = activation_info.activation();

                    BI_COMPUTE_RETURN_ERROR_ON_MSG(
                        is_data_type_quantized_asymmetric_char(data_type) && oq_info.is_dynamic() &&
                        (std::find(std::begin(qasymm8_static_quant_activations),
                            std::end(qasymm8_static_quant_activations),
                            f_act) == std::end(qasymm8_static_quant_activations)),
                        "For QASYMM8 statically quantized, only relu and lower/upper bounded relu are supported");

                    BI_COMPUTE_RETURN_ERROR_ON_MSG(
                        is_data_type_quantized_asymmetric(data_type) &&
                        (std::find(std::begin(qasymm8_activations), std::end(qasymm8_activations), f_act) ==
                            std::end(qasymm8_activations)),
                        "For QASYMM8 only hard swish, leaky relu, tanh, logistic, relu and lower/upper bounded relu are supported")
                    ;

                    BI_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized_symmetric(data_type) &&
                                                   (std::find(std::begin(qsymm16_activations),
                                                       std::end(qsymm16_activations),
                                                       f_act) == std::end(qsymm16_activations)),
                                                   "For QSYMM16 only tanh and logistic are supported");
                    BI_COMPUTE_RETURN_ERROR_ON(
                        (data_type == BIDataType::QASYMM8 || data_type == BIDataType::QASYMM16) &&
                        (f_act == BIActivationLayerInfo::ActivationFunction::TANH) &&
                        (oq_info != BIQuantizationInfo(1.f / 128.f, 128)));
                    BI_COMPUTE_RETURN_ERROR_ON(
                        (data_type == BIDataType::QASYMM8 || data_type == BIDataType::QASYMM16) &&
                        (f_act == BIActivationLayerInfo::ActivationFunction::LOGISTIC) &&
                        (oq_info != BIQuantizationInfo(1.f / 256.f, 0)));

                    BI_COMPUTE_RETURN_ERROR_ON(data_type == BIDataType::QASYMM8_SIGNED &&
                        (f_act == BIActivationLayerInfo::ActivationFunction::TANH) &&
                        (oq_info != BIQuantizationInfo(1.f / 128.f, 0)));
                    BI_COMPUTE_RETURN_ERROR_ON(data_type == BIDataType::QASYMM8_SIGNED &&
                        (f_act == BIActivationLayerInfo::ActivationFunction::LOGISTIC) &&
                        (oq_info != BIQuantizationInfo(1.f / 256.f, -128)));

                    BI_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) &&
                        (f_act == BIActivationLayerInfo::ActivationFunction::TANH) &&
                        (oq_info != BIQuantizationInfo(1.f / 32768.f, 0)));
                    BI_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) &&
                        (f_act == BIActivationLayerInfo::ActivationFunction::LOGISTIC) &&
                        (oq_info != BIQuantizationInfo(1.f / 32768.f, 0)));

                    // Checks performed when dst is configured
                    if ((dst != nullptr) && (dst->total_size() != 0)) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                    }

                    return BIStatus{};
                }

#ifdef __aarch64__

                // TODO (COMPMID-7511): delegate to LUTManager
                void init_lut(BIActivationLayerInfo::ActivationFunction act_func,
                              BIDataType data_type,
                              const BIUniformQuantizationInfo &qi_in,
                              const BIUniformQuantizationInfo &qi_out,
                              BIActivationLayerInfo::LookupTable256 &lut,
                              float a,
                              float b) {
                    for (size_t i = 0; i < lut.size(); ++i) {
                        float tmp_f =
                                (data_type == BIDataType::QASYMM8)
                                    ? dequantize_qasymm8(i, qi_in)
                                    : dequantize_qasymm8_signed(i, qi_in);
                        switch (act_func) {
                            case BIActivationLayerInfo::ActivationFunction::HARD_SWISH:
                                tmp_f = tmp_f * ((std::min(std::max((tmp_f + 3), 0.0f), 6.0f)) * 0.166666667f);
                                break;
                            case BIActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                                tmp_f = tmp_f > 0 ? tmp_f : tmp_f * a;
                                break;
                            case BIActivationLayerInfo::ActivationFunction::LOGISTIC:
                                tmp_f = 1.f / (1.f + std::exp(-tmp_f));
                                break;
                            case BIActivationLayerInfo::ActivationFunction::ABS:
                                tmp_f = std::abs(tmp_f);
                                break;
                            case BIActivationLayerInfo::ActivationFunction::LINEAR:
                                tmp_f = a * tmp_f + b;
                                break;
                            case BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                                tmp_f = std::min<>(a, std::max(0.f, tmp_f));
                                break;
                            case BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                                tmp_f = std::min<>(a, std::max<>(b, tmp_f));
                                break;
                            case BIActivationLayerInfo::ActivationFunction::SOFT_RELU:
                                tmp_f = (tmp_f > 12.f) ? tmp_f : std::log(1.f + std::exp(tmp_f));
                                break;
                            case BIActivationLayerInfo::ActivationFunction::ELU:
                                tmp_f = (tmp_f >= 0) ? tmp_f : a * (std::exp(tmp_f) - 1);
                                break;
                            case BIActivationLayerInfo::ActivationFunction::SQRT:
                                tmp_f = std::sqrt(tmp_f);
                                break;
                            case BIActivationLayerInfo::ActivationFunction::SQUARE:
                                tmp_f = tmp_f * tmp_f;
                                break;
                            case BIActivationLayerInfo::ActivationFunction::TANH:
                                tmp_f = a * std::tanh(b * tmp_f);
                                break;
                            case BIActivationLayerInfo::ActivationFunction::IDENTITY:
                                break;
                            case BIActivationLayerInfo::ActivationFunction::SWISH:
                                tmp_f = tmp_f / (1.f + std::exp(-a * tmp_f));
                                break;
                            case BIActivationLayerInfo::ActivationFunction::GELU:
                                // 0.5 * x(1 + erf(x / 2^(1/2))
                                tmp_f = tmp_f * (0.5f * (1.0f + erff(tmp_f / 1.41421356237f)));
                                break;
                            default:
                                BI_COMPUTE_ERROR("Not supported");
                                tmp_f = 0;
                                break;
                        }
                        lut[i] =
                                (data_type == BIDataType::QASYMM8)
                                    ? quantize_qasymm8(tmp_f, qi_out)
                                    : quantize_qasymm8_signed(tmp_f, qi_out);
                    }
                }

#endif // __aarch64__
            }

            void BICpuActivationKernel::configure(const BIITensorInfo *src,
                                                  BIITensorInfo *dst,
                                                  BIActivationLayerInfo activation_info) {
                BI_COMPUTE_UNUSED(dst);
                BI_COMPUTE_ERROR_ON_NULLPTR(src);
                BI_COMPUTE_ERROR_THROW_ON(BICpuActivationKernel::validate(src, dst, activation_info));

                heuristics::BICpuActivationKernelHeuristics heuristics(src, dst, activation_info);
                _heuristics = std::move(heuristics);

                if (dst != nullptr) {
                    // dst auto inizialitation if not yet initialized
                    auto_init_if_empty(*dst, *src->clone());
                }

                const auto *uk = heuristics.kernel();
                BI_COMPUTE_ERROR_ON_NULLPTR(uk);

                _name = std::string("BICpuActivationKernel").append("/").append(uk->name);

#ifdef __aarch64__
                // 初始化查找表
                BILUTManager &lut_manager = BILUTManager::get_instance();

                // TODO (COMPMID-7511): delegate to LUTManager
                if ((src->data_type() == BIDataType::QASYMM8 || src->data_type() == BIDataType::QASYMM8_SIGNED) &&
                    activation_info.activation() != BIActivationFunction::RELU) {
                    BIActivationLayerInfo::LookupTable256 tmp_lut;
                    init_lut(activation_info.activation(), src->data_type(), src->quantization_info().uniform(),
                             (dst) ? dst->quantization_info().uniform() : src->quantization_info().uniform(), tmp_lut,
                             activation_info.a(), activation_info.b());
                    activation_info.setLookupTable256(tmp_lut);
                }

                if (std::string(uk->name) == "sve_fp16_activation_lut") {
                    // Create info using init list.
                    const BILUTInfo info = {
                        activation_info.activation(), activation_info.a(), activation_info.b(),
                        src->data_type(),
                        src->quantization_info().uniform()
                    };
                    activation_info.setLookupTable65536((lut_manager.get_lut_table<LookupTable65536>(info)));
                }
#endif // __aarch64__
                _act_info = activation_info;

                BIICPPKernel::configure(heuristics.window());
            }


            BIStatus
            BICpuActivationKernel::validate(const BIITensorInfo *src, const BIITensorInfo *dst,

                                            const BIActivationLayerInfo &act_info) {
                BI_COMPUTE_UNUSED(act_info);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, act_info));

                return BIStatus{};
            }

            void BICpuActivationKernel::dynamic_change_win(const BIITensorInfo *src) {
                _heuristics.dynamic_change_win(src);

                BIICPPKernel::dynamic_configure(_heuristics.window());
            }


            size_t BICpuActivationKernel::get_mws(const CPUInfo &platform, size_t thread_count) const {
                BI_COMPUTE_UNUSED(thread_count);
                BI_COMPUTE_UNUSED(platform);

                return _heuristics.mws();
            }

            void BICpuActivationKernel::run_op(BIITensorPack &tensors,
                                               const BIWindow &window,
                                               const ThreadInfo &info) {
                // Early exit on disabled activation
                if (!_act_info.enabled()) {
                    return;
                }

                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIIKernel::window(), window);

                BI_COMPUTE_ERROR_ON(tensors.empty());

                BIActivationKernelPtr run_method = _heuristics.kernel()->ukernel;
                BI_COMPUTE_ERROR_ON(run_method == nullptr);

                const BIITensor *src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                BIITensor *dst = tensors.get_tensor(BITensorType::ACL_DST);

                run_method(src, dst, _act_info, window);
            }

            const char *BICpuActivationKernel::name() const {
                return _name.c_str();
            }
        }
    }
}
