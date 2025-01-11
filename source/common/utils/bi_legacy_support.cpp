//
// Created by Mason on 2025/1/11.
//

#include <common/utils/legacy_support.hpp>

#include <function_info/bi_activationLayerInfo.h>

namespace BatmanInfer {
    namespace detail {
        namespace {
            BIDataType convert_to_legacy_data_type(BclDataType data_type) {
                switch (data_type) {
                    case BclDataType::BclFloat32:
                        return BIDataType::F32;
                    case BclDataType::BclFloat16:
                        return BIDataType::F16;
                    case BclDataType::BclBFloat16:
                        return BIDataType::BFLOAT16;
                    default:
                        return BIDataType::UNKNOWN;
                }
            }

            BclDataType convert_to_c_data_type(BIDataType data_type) {
                switch (data_type) {
                    case BIDataType::F32:
                        return BclDataType::BclFloat32;
                    case BIDataType::F16:
                        return BclDataType::BclFloat16;
                    case BIDataType::BFLOAT16:
                        return BclDataType::BclBFloat16;
                    default:
                        return BclDataType::BclDataTypeUnknown;
                }
            }

            BITensorShape create_legacy_tensor_shape(int32_t ndims, int32_t *shape) {
                BITensorShape legacy_shape{};
                for (int32_t d = 0; d < ndims; ++d) {
                    legacy_shape.set(d, shape[d], false);
                }
                return legacy_shape;
            }

            int32_t *create_tensor_shape_array(const BITensorInfo &info) {
                const auto num_dims = info.num_dimensions();
                if (num_dims <= 0) {
                    return nullptr;
                }

                int32_t *shape_array = new int32_t[num_dims];

                for (size_t d = 0; d < num_dims; ++d) {
                    shape_array[d] = info.tensor_shape()[d];
                }

                return shape_array;
            }
        } // namespace

        BITensorInfo convert_to_legacy_tensor_info(const BclTensorDescriptor &desc) {
            BITensorInfo legacy_desc;
            legacy_desc.init(create_legacy_tensor_shape(desc.ndims, desc.shape), 1,
                             convert_to_legacy_data_type(desc.data_type));
            return legacy_desc;
        }

        BclTensorDescriptor convert_to_descriptor(const BITensorInfo &info) {
            const auto num_dims = info.num_dimensions();
            BclTensorDescriptor desc{static_cast<int32_t>(num_dims), create_tensor_shape_array(info),
                                     convert_to_c_data_type(info.data_type()), nullptr, 0};
            return desc;
        }

        BIActivationLayerInfo convert_to_activation_info(const BclActivationDescriptor &desc) {
            BIActivationLayerInfo::ActivationFunction act;
            switch (desc.type) {
                case BclActivationType::BclIdentity:
                    act = BIActivationLayerInfo::ActivationFunction::IDENTITY;
                    break;
                case BclActivationType::BclLogistic:
                    act = BIActivationLayerInfo::ActivationFunction::LOGISTIC;
                    break;
                case BclActivationType::BclTanh:
                    act = BIActivationLayerInfo::ActivationFunction::TANH;
                    break;
                case BclActivationType::BclRelu:
                    act = BIActivationLayerInfo::ActivationFunction::RELU;
                    break;
                case BclActivationType::BclBoundedRelu:
                    act = BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU;
                    break;
                case BclActivationType::BclLuBoundedRelu:
                    act = BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU;
                    break;
                case BclActivationType::BclLeakyRelu:
                    act = BIActivationLayerInfo::ActivationFunction::LEAKY_RELU;
                    break;
                case BclActivationType::BclSoftRelu:
                    act = BIActivationLayerInfo::ActivationFunction::SOFT_RELU;
                    break;
                case BclActivationType::BclElu:
                    act = BIActivationLayerInfo::ActivationFunction::ELU;
                    break;
                case BclActivationType::BclAbs:
                    act = BIActivationLayerInfo::ActivationFunction::ABS;
                    break;
                case BclActivationType::BclSquare:
                    act = BIActivationLayerInfo::ActivationFunction::SQUARE;
                    break;
                case BclActivationType::BclSqrt:
                    act = BIActivationLayerInfo::ActivationFunction::SQRT;
                    break;
                case BclActivationType::BclLinear:
                    act = BIActivationLayerInfo::ActivationFunction::LINEAR;
                    break;
                case BclActivationType::BclHardSwish:
                    act = BIActivationLayerInfo::ActivationFunction::HARD_SWISH;
                    break;
                default:
                    return BIActivationLayerInfo();
            }

            return BIActivationLayerInfo(act, desc.a, desc.b);
        }
    }
}