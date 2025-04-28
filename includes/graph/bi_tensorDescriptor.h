//
// Created by holynova on 2025/1/7.
//

#ifndef BATMANINFER_GRAPH_BI_TENSORDESCRIPTOR_H
#define BATMANINFER_GRAPH_BI_TENSORDESCRIPTOR_H

#include "graph/bi_types.h"
#include "batman_compute/support/i_clone_able.h"

namespace BatmanInfer {

namespace graph {

    /** Tensor metadata class */
    struct BITensorDescriptor final : public misc::ICloneable<BITensorDescriptor>
    {
        /** Default Constructor **/
        BITensorDescriptor() = default;
        /** Constructor
         *
         * @param[in] tensor_shape       Tensor shape
         * @param[in] tensor_data_type   Tensor data type
         * @param[in] tensor_quant_info  Tensor quantization info
         * @param[in] tensor_data_layout Tensor data layout
         * @param[in] tensor_target      Target to allocate the tensor for
         */
        BITensorDescriptor(BITensorShape      tensor_shape,
                           BIDataType         tensor_data_type,
                           BIQuantizationInfo tensor_quant_info  = BIQuantizationInfo(),
                           BIDataLayout       tensor_data_layout = BIDataLayout::NCHW,
                           BITarget           tensor_target      = BITarget::UNSPECIFIED)
            : shape(tensor_shape),
              data_type(tensor_data_type),
              layout(tensor_data_layout),
              quant_info(tensor_quant_info),
              target(tensor_target)
        {
        }
        /** Sets tensor descriptor shape
         *
         * @param[in] tensor_shape Tensor shape to set
         *
         * @return This tensor descriptor
         */
        BITensorDescriptor &set_shape(BITensorShape &tensor_shape)
        {
            shape = tensor_shape;
            return *this;
        }
        /** Sets tensor descriptor data type
         *
         * @param[in] tensor_data_type Data type
         *
         * @return This tensor descriptor
         */
        BITensorDescriptor &set_data_type(BIDataType tensor_data_type)
        {
            data_type = tensor_data_type;
            return *this;
        }
        /** Sets tensor descriptor data layout
         *
         * @param[in] data_layout Data layout
         *
         * @return This tensor descriptor
         */
        BITensorDescriptor &set_layout(BIDataLayout data_layout)
        {
            layout = data_layout;
            return *this;
        }
        /** Sets tensor descriptor quantization info
         *
         * @param[in] tensor_quant_info Quantization information
         *
         * @return This tensor descriptor
         */
        BITensorDescriptor &set_quantization_info(BIQuantizationInfo tensor_quant_info)
        {
            quant_info = tensor_quant_info;
            return *this;
        }

        // Inherited methods overridden:
        std::unique_ptr<BITensorDescriptor> clone() const override
        {
            return std::make_unique<BITensorDescriptor>(*this);
        }

        BITensorShape      shape{};                        /**< Tensor shape */
        BIDataType         data_type{BIDataType::UNKNOWN}; /**< Data type */
        BIDataLayout       layout{BIDataLayout::NCHW};     /**< Data layout */
        BIQuantizationInfo quant_info{};                   /**< Quantization info */
        BITarget           target{BITarget::UNSPECIFIED};  /**< Target */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_TENSORDESCRIPTOR_H
