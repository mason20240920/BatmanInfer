//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_PRINTLAYERNODE_H
#define BATMANINFER_GRAPH_PRINTLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

// Forward declarations
class BIITensor;

namespace graph {

    /** Print Layer node */
    class PrintLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] stream      Output stream.
         * @param[in] format_info (Optional) Format info.
         * @param[in] transform   (Optional) Input transform function.
         */
        PrintLayerNode(std::ostream                                 &stream,
                       const BIIOFormatInfo                         &format_info = BIIOFormatInfo(),
                       const std::function<BIITensor *(BIITensor *)> transform   = nullptr);

        /** Stream metadata accessor
         *
         * @return Print Layer stream
         */
        std::ostream &stream() const;

        /** Formatting metadata accessor
         *
         * @return Print Layer format info
         */
        const BIIOFormatInfo format_info() const;

        /** Transform function metadata accessor
         *
         * @return Print Layer transform function
         */
        const std::function<BIITensor *(BIITensor *)> transform() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        std::ostream                                 &_stream;
        const BIIOFormatInfo                          _format_info;
        const std::function<BIITensor *(BIITensor *)> _transform;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_PRINTLAYERNODE_H
