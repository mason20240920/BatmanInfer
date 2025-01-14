//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/OutputNode.h"

#include"data/core/bi_error.h"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_tensor.h"

namespace BatmanInfer {

namespace graph {

    OutputNode::OutputNode()
    {
        _input_edges.resize(1, EmptyEdgeID);
    }

    bool OutputNode::forward_descriptors()
    {
        return true;
    }

    BITensorDescriptor OutputNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        return BITensorDescriptor();
    }

    BINodeType OutputNode::type() const
    {
        return BINodeType::Output;
    }

    void OutputNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
