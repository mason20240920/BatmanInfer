//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_DETECTIONOUTPUTLAYERNODE_H
#define BATMANINFER_GRAPH_DETECTIONOUTPUTLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** DetectionOutput Layer node */
    class DetectionOutputLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] detection_info DetectionOutput Layer information
         */
        DetectionOutputLayerNode(BIDetectionOutputLayerInfo detection_info);
        /** DetectionOutput metadata accessor
         *
         * @return DetectionOutput Layer info
         */
        BIDetectionOutputLayerInfo detection_output_info() const;
        /** Computes detection output output descriptor
         *
         * @param[in] input_descriptor Input descriptor
         * @param[in] info             DetectionOutput operation attributes
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor         &input_descriptor,
                                                            const BIDetectionOutputLayerInfo &info);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIDetectionOutputLayerInfo _info;

        // Each detection contains a bounding box, given by its coordinates xmin, ymin, xmax, ymax, associated at the respective image, label and a confidence
        static const int detection_size = 7;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_DETECTIONOUTPUTLAYERNODE_H
