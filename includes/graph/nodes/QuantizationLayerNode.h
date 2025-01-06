//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_QUANTIZATIONLAYERNODE_H
#define BATMANINFER_GRAPH_QUANTIZATIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Quantization Layer node */
    class QuantizationLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] out_quant_info Output quantization info
         */
        QuantizationLayerNode(BIQuantizationInfo out_quant_info);

        /** Constructor
         *
         * @param[in] out_quant_info Output quantization info
         * @param[in] out_data_type  Output data type
         */
        QuantizationLayerNode(BIQuantizationInfo out_quant_info, BIDataType out_data_type);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

        static constexpr BINodeType node_type = BINodeType::QuantizationLayer;

    private:
        BIQuantizationInfo _out_quant_info;
        BIDataType         _out_data_type;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_QUANTIZATIONLAYERNODE_H
