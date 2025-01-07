//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_DETECTIONPOSTPROCESSLAYERNODE_H
#define BATMANINFER_GRAPH_DETECTIONPOSTPROCESSLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** DetectionPostProcess Layer node */
    class DetectionPostProcessLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] detection_info DetectionPostProcess Layer information
         */
        DetectionPostProcessLayerNode(BIDetectionPostProcessLayerInfo detection_info);
        /** DetectionPostProcess metadata accessor
         *
         * @return DetectionPostProcess Layer info
         */
        BIDetectionPostProcessLayerInfo detection_post_process_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIDetectionPostProcessLayerInfo _info;

        static const int kNumCoordBox = 4;
        static const int kBatchSize   = 1;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_DETECTIONPOSTPROCESSLAYERNODE_H
