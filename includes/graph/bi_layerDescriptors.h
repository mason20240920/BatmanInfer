//
// Created by holynova on 2025/1/2.
//

#ifndef BATMANINFER_BI_LAYERDESCRIPTORS_H
#define BATMANINFER_BI_LAYERDESCRIPTORS_H

#include "data/core/bi_types.hpp"
#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

namespace descriptors {

    /** Concatenate layer descriptor */
    struct BIConcatLayerDescriptor
    {
        /** Default constructor */
        BIConcatLayerDescriptor() : axis(BIDataLayoutDimension::CHANNEL), output_qinfo()
        {
        }

        /** Constructor concatenate layer descriptor
         *
         * @param[in] axis Axis.
         */
        BIConcatLayerDescriptor(BIDataLayoutDimension axis) : axis(axis), output_qinfo()
        {
        }

        /** Constructor concatenate layer descriptor
         *
         * @param[in] axis         Axis.
         * @param[in] output_qinfo Output quantization info.
         */
        BIConcatLayerDescriptor(BIDataLayoutDimension axis, BIQuantizationInfo output_qinfo)
            : axis(axis), output_qinfo(output_qinfo)
        {
        }

        const BIDataLayoutDimension axis;         /**< Concatenation Axis */
        const BIQuantizationInfo    output_qinfo; /**< Output quantizazion info */
    };

    /** Elementwise layer descriptor */
    struct EltwiseLayerDescriptor
    {
        /** Constructor
         *
         * @param[in] op               Element-wise operation to perform
         * @param[in] out_quant_info   (Optional) Output quantization information. Defaults to empty @ref QuantizationInfo
         * @param[in] c_policy         (Optional) Convert policy used for the operation. Defaults to @ref ConvertPolicy::SATURATE
         * @param[in] r_policy         (Optional) Rounding policy used for the operation. Defaults to @ref RoundingPolicy::TO_ZERO
         * @param[in] fused_activation (Optional) Fused activation information. Defaults to empty (identity) @ref ActivationLayerInfo
         */
        EltwiseLayerDescriptor(BIEltwiseOperation    op,
                               BIQuantizationInfo    out_quant_info   = BIQuantizationInfo(),
                               BIConvertPolicy       c_policy         = BIConvertPolicy::SATURATE,
                               BIRoundingPolicy      r_policy         = BIRoundingPolicy::TO_ZERO,
                               BIActivationLayerInfo fused_activation = BIActivationLayerInfo())
            : op(op),
              out_quant_info(out_quant_info),
              c_policy(c_policy),
              r_policy(r_policy),
              fused_activation(fused_activation)
        {
        }

        BIEltwiseOperation    op;               /**< Element-wise operation to perform */
        BIQuantizationInfo    out_quant_info;   /**< Output quantization information */
        BIConvertPolicy       c_policy;         /**< Convert policy */
        BIRoundingPolicy      r_policy;         /**< Rounding policy */
        BIActivationLayerInfo fused_activation; /**< Fused activation info */
    };

    /** Unary Elementwise layer descriptor */
    struct UnaryEltwiseLayerDescriptor
    {
        /** Constructor
         *
         * @param[in] op               Unary element-wise operation to perform
         * @param[in] out_quant_info   (Optional) Output quantization information. Defaults to empty @ref QuantizationInfo
         * @param[in] c_policy         (Optional) Convert policy used for the operation. Defaults to @ref ConvertPolicy::SATURATE
         * @param[in] r_policy         (Optional) Rounding policy used for the operation. Defaults to @ref RoundingPolicy::TO_ZERO
         * @param[in] fused_activation (Optional) Fused activation information. Defaults to empty (identity) @ref ActivationLayerInfo
         */
        UnaryEltwiseLayerDescriptor(BIUnaryEltwiseOperation op,
                                    BIQuantizationInfo      out_quant_info   = BIQuantizationInfo(),
                                    BIConvertPolicy         c_policy         = BIConvertPolicy::SATURATE,
                                    BIRoundingPolicy        r_policy         = BIRoundingPolicy::TO_ZERO,
                                    BIActivationLayerInfo   fused_activation = BIActivationLayerInfo())
            : op(op),
              out_quant_info(out_quant_info),
              c_policy(c_policy),
              r_policy(r_policy),
              fused_activation(fused_activation)
        {
        }

        BIUnaryEltwiseOperation op;               /**< Unary element-wise operation to perform */
        BIQuantizationInfo      out_quant_info;   /**< Output quantization information */
        BIConvertPolicy         c_policy;         /**< Convert policy */
        BIRoundingPolicy        r_policy;         /**< Rounding policy */
        BIActivationLayerInfo   fused_activation; /**< Fused activation info */
    };

    /** Deconvolution layer descriptor */
    struct DeconvolutionLayerDescriptor
    {
        /** Constructor
         *
         * @param[in] info           Dedonvolution layer attributes
         * @param[in] out_quant_info (Optional) Output quantization infomation
         */
        DeconvolutionLayerDescriptor(BIPadStrideInfo info, BIQuantizationInfo out_quant_info = BIQuantizationInfo())
            : info(info), out_quant_info(out_quant_info)
        {
        }

        BIPadStrideInfo    info;           /**< Padding and stride information */
        BIQuantizationInfo out_quant_info; /**< Output quantization information */
    };

} // namespace descriptors

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_BI_LAYERDESCRIPTORS_H
