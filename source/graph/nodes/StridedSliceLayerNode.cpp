//
// Created by holynova on 2025/1/15.
//

#include "graph/nodes/StridedSliceLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    StridedSliceLayerNode::StridedSliceLayerNode(const BICoordinates    &starts,
                                                 const BICoordinates    &ends,
                                                 const BiStrides        &strides,
                                                 BIStridedSliceLayerInfo info)
        : _starts(starts), _ends(ends), _strides(strides), _info(std::move(info))
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BICoordinates StridedSliceLayerNode::starts() const
    {
        return _starts;
    }

    BICoordinates StridedSliceLayerNode::ends() const
    {
        return _ends;
    }

    BiStrides StridedSliceLayerNode::strides() const
    {
        return _strides;
    }

    BIStridedSliceLayerInfo StridedSliceLayerNode::strided_slice_info() const
    {
        return _info;
    }

    BITensorDescriptor StridedSliceLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                                        const BICoordinates      &starts,
                                                                        const BICoordinates      &ends,
                                                                        const BiStrides          &strides,
                                                                        BIStridedSliceLayerInfo   info)
    {
        using namespace BatmanInfer::helpers::tensor_transform;

        BITensorDescriptor output_desc = input_descriptor;
        output_desc.shape             = compute_strided_slice_output_shape(input_descriptor.shape, starts, ends, strides,
                                                                          info.begin_mask(), info.end_mask(), info.shrink_axis_mask());

        return output_desc;
    }

    bool StridedSliceLayerNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor StridedSliceLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return compute_output_descriptor(src->desc(), _starts, _ends, _strides, _info);
    }

    BINodeType StridedSliceLayerNode::type() const
    {
        return BINodeType::StridedSliceLayer;
    }

    void StridedSliceLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
