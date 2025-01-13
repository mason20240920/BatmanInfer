//
// Created by Mason on 2025/1/13.
//

#include <data/core/utils/bi_assembly_utils.hpp>

#include <function_info/bi_activationLayerInfo.h>

namespace BatmanInfer {
    namespace assembly_utils {
        BatmanGemm::Activation map_to_batman_gemm_activation(const BIActivationLayerInfo &act) {
            BatmanGemm::Activation gemm_act;
            // Early exit in case lower bound is other than 0, as it's not yet supported
            if (act.b() != 0.f) {
                return gemm_act;
            }

            switch (act.activation()) {
                case BIActivationLayerInfo::ActivationFunction::RELU:
                    gemm_act.type = BatmanGemm::Activation::Type::ReLU;
                    break;
                case BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                    gemm_act.type = BatmanGemm::Activation::Type::BoundedReLU;
                    gemm_act.param1 = act.a();
                    gemm_act.param2 = 0.f;
                    break;
                case BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                    gemm_act.type = BatmanGemm::Activation::Type::BoundedReLU;
                    gemm_act.param1 = act.a();
                    gemm_act.param2 = act.b();
                    break;
                default:
                    gemm_act.type = BatmanGemm::Activation::Type::None;
            }

            return gemm_act;
        }

        BatmanConv::PaddingValues map_to_batman_conv_padding(const BIPadStrideInfo &pad_stride_info) {
            return BatmanConv::PaddingValues{pad_stride_info.pad_left(),
                                             pad_stride_info.pad_top(),
                                             pad_stride_info.pad_right(),
                                             pad_stride_info.pad_bottom()};
        }

        BatmanGemm::WeightFormat map_to_batman_gemm_weight_format(const BatmanInfer::BIWeightFormat &weight_format) {
            BatmanGemm::WeightFormat gemm_weight_format;

            switch (weight_format) {
                case BatmanInfer::BIWeightFormat::UNSPECIFIED:
                    gemm_weight_format = BatmanGemm::WeightFormat::UNSPECIFIED;
                    break;
                case BatmanInfer::BIWeightFormat::ANY:
                    gemm_weight_format = BatmanGemm::WeightFormat::ANY;
                    break;
                case BatmanInfer::BIWeightFormat::OHWI:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWI;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo2:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo2;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo4:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo4;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo8;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo32:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo32;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo64:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo64;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo128:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo128;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo4i2:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo4i2;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo4i2_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo4i2_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo8i2:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo8i2;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo8i2_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo8i2_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo16i2:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo16i2;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo16i2_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo16i2_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo32i2:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo32i2;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo32i2_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo32i2_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo64i2:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo64i2;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo64i2_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo64i2_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo4i4:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo4i4;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo4i4_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo4i4_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo8i4:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo8i4;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo8i4_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo8i4_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo16i4:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo16i4;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo16i4_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo16i4_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo32i4:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo32i4;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo32i4_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo32i4_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo64i4:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo64i4;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo64i4_bf16:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo64i4_bf16;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo2i8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo2i8;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo4i8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo4i8;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo8i8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo8i8;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo16i8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo16i8;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo32i8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo32i8;
                    break;
                case BatmanInfer::BIWeightFormat::OHWIo64i8:
                    gemm_weight_format = BatmanGemm::WeightFormat::OHWIo64i8;
                    break;
                default:
                    gemm_weight_format = BatmanGemm::WeightFormat::UNSPECIFIED;
            }
            return gemm_weight_format;
        }

        BatmanInfer::BIWeightFormat map_to_batman_compute_weight_format(const BatmanGemm::WeightFormat &weight_format) {
            BatmanInfer::BIWeightFormat bcl_weight_format;

            switch (weight_format) {
                case BatmanGemm::WeightFormat::UNSPECIFIED:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::UNSPECIFIED;
                    break;
                case BatmanGemm::WeightFormat::ANY:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::ANY;
                    break;
                case BatmanGemm::WeightFormat::OHWI:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWI;
                    break;
                case BatmanGemm::WeightFormat::OHWIo2:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo2;
                    break;
                case BatmanGemm::WeightFormat::OHWIo4:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo4;
                    break;
                case BatmanGemm::WeightFormat::OHWIo8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo8;
                    break;
                case BatmanGemm::WeightFormat::OHWIo16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo32:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo32;
                    break;
                case BatmanGemm::WeightFormat::OHWIo64:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo64;
                    break;
                case BatmanGemm::WeightFormat::OHWIo128:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo128;
                    break;
                case BatmanGemm::WeightFormat::OHWIo4i2:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo4i2;
                    break;
                case BatmanGemm::WeightFormat::OHWIo4i2_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo4i2_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo8i2:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo8i2;
                    break;
                case BatmanGemm::WeightFormat::OHWIo8i2_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo8i2_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo16i2:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo16i2;
                    break;
                case BatmanGemm::WeightFormat::OHWIo16i2_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo16i2_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo32i2:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo32i2;
                    break;
                case BatmanGemm::WeightFormat::OHWIo32i2_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo32i2_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo64i2:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo64i2;
                    break;
                case BatmanGemm::WeightFormat::OHWIo64i2_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo64i2_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo4i4:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo4i4;
                    break;
                case BatmanGemm::WeightFormat::OHWIo4i4_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo4i4_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo8i4:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo8i4;
                    break;
                case BatmanGemm::WeightFormat::OHWIo8i4_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo8i4_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo16i4:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo16i4;
                    break;
                case BatmanGemm::WeightFormat::OHWIo16i4_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo16i4_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo32i4:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo32i4;
                    break;
                case BatmanGemm::WeightFormat::OHWIo32i4_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo32i4_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo64i4:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo64i4;
                    break;
                case BatmanGemm::WeightFormat::OHWIo64i4_bf16:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo64i4_bf16;
                    break;
                case BatmanGemm::WeightFormat::OHWIo2i8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo2i8;
                    break;
                case BatmanGemm::WeightFormat::OHWIo4i8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo4i8;
                    break;
                case BatmanGemm::WeightFormat::OHWIo8i8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo8i8;
                    break;
                case BatmanGemm::WeightFormat::OHWIo16i8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo16i8;
                    break;
                case BatmanGemm::WeightFormat::OHWIo32i8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo32i8;
                    break;
                case BatmanGemm::WeightFormat::OHWIo64i8:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::OHWIo64i8;
                    break;
                default:
                    bcl_weight_format = BatmanInfer::BIWeightFormat::UNSPECIFIED;
            }
            return bcl_weight_format;
        }
    } // namespace assembly_utils
}