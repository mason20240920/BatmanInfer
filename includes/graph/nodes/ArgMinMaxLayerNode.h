//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_ARGMINMAXLAYERNODE_H
#define BATMANINFER_GRAPH_ARGMINMAXLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Arg Min/Max Layer node */
    class ArgMinMaxLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] op             Operation to perform: min or max
         * @param[in] axis           Axis along which to reduce. Supported reduction axis : 0,1,2,3
         * @param[in] out_data_type  (Optional) Output data type
         * @param[in] out_quant_info (Optional) Output quantization info
         */
        ArgMinMaxLayerNode(BIReductionOperation op,
                           unsigned int         axis,
                           BIDataType           out_data_type  = BIDataType::UNKNOWN,
                           BIQuantizationInfo   out_quant_info = BIQuantizationInfo());
        /** Operator accessor
         *
         * @return The operator the layer performs: min or max
         */
        BIReductionOperation reduction_operation() const;
        /** Axis accessor
         *
         * @return The axis along which the reduction is operating
         */
        unsigned int axis() const;
        /** Output data type accessor
         *
         * @return The output data type
         */
        BIDataType out_data_type() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    public:
        static constexpr BINodeType node_type = BINodeType::ArgMinMaxLayer;

    private:
        BIReductionOperation _op;
        unsigned int         _axis;
        BIDataType           _out_data_type;
        BIQuantizationInfo   _out_quant_info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_ARGMINMAXLAYERNODE_H
