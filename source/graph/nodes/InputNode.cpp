//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/InputNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    InputNode::InputNode(BITensorDescriptor desc) : _desc(std::move(desc))
    {
        _outputs.resize(1, NullTensorID);
    }

    bool InputNode::forward_descriptors()
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

    BITensorDescriptor InputNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        return _desc;
    }

    BINodeType InputNode::type() const
    {
        return BINodeType::Input;
    }

    void InputNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
