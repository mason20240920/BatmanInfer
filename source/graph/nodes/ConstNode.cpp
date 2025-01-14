//
// Created by holynova on 2025/1/13.
//

#include "graph/nodes/ConstNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    ConstNode::ConstNode(BITensorDescriptor desc) : _desc(std::move(desc))
    {
        _outputs.resize(1, NullTensorID);
    }

    bool ConstNode::forward_descriptors()
    {
        if (output_id(0) != NullTensorID)
        {
            BITensor *t = output(0);
            BI_COMPUTE_ERROR_ON(t == nullptr);
            t->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor ConstNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        return _desc;
    }

    BINodeType ConstNode::type() const
    {
        return BINodeType::Const;
    }

    void ConstNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
