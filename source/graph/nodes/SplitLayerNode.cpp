//
// Created by holynova on 2025/1/15.
//

#include "graph/nodes/SplitLayerNode.h"

#include "data/core/bi_helpers.hpp"
#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    SplitLayerNode::SplitLayerNode(unsigned int num_splits, int axis, std::vector<int> size_splits)
    : _num_splits(num_splits), _axis(axis), _size_splits(size_splits)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(num_splits, NullTensorID);
    }

    unsigned int SplitLayerNode::num_splits() const
    {
        return _num_splits;
    }

    unsigned int SplitLayerNode::axis() const
    {
        return _axis;
    }

    std::pair<BITensorDescriptor, BICoordinates> SplitLayerNode::compute_output_descriptor(
        const BITensorDescriptor &input_descriptor, unsigned int num_splits, int axis, unsigned int idx)
    {
        // Handle negative axis, negative index is used to specify axis from the end (e.g. -1 for the last axis).
        int                num_dimension = static_cast<int32_t>(input_descriptor.shape.num_dimensions());
        int                tmp_axis      = wrap_around(axis, num_dimension);
        BICoordinates      coords;
        BITensorDescriptor output_descriptor = input_descriptor;
        int                split_size        = input_descriptor.shape[tmp_axis] / num_splits;
        if (_size_splits.empty())
        {
            output_descriptor.shape.set(tmp_axis, split_size);
            coords.set(tmp_axis, idx * split_size);
        }
        else
        {
            int split_size = _size_splits[idx];
            if (split_size == -1)
            {
                split_size = input_descriptor.shape[tmp_axis];
                for (unsigned int i = 0; i < _size_splits.size() - 1; ++i)
                    split_size -= _size_splits[i];
            }
            output_descriptor.shape.set(tmp_axis, split_size);
            int coord_value = 0;
            for (unsigned int i = 0; i < idx; ++i)
                coord_value += _size_splits[i];
            coords.set(tmp_axis, coord_value);
        }

        return std::make_pair(output_descriptor, coords);
    }

    bool SplitLayerNode::forward_descriptors()
    {
        if (input_id(0) != NullTensorID)
        {
            validate();
            for (unsigned int i = 0; i < _outputs.size(); ++i)
            {
                if (output_id(i) != NullTensorID)
                {
                    BITensor *dst_i = output(i);
                    BI_COMPUTE_ERROR_ON(dst_i == nullptr);
                    dst_i->desc() = configure_output(i);
                }
            }
            return true;
        }
        return false;
    }

    BITensorDescriptor SplitLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor input_descriptor  = src->desc();
        BITensorDescriptor output_descriptor = input_descriptor;

        // Handle negative axis, negative index is used to specify axis from the end (e.g. -1 for the last axis).
        int num_dimension = static_cast<int32_t>(src->desc().shape.num_dimensions());
        int tmp_axis      = wrap_around(_axis, num_dimension);

        int split_size = (_size_splits.empty()) ? (input_descriptor.shape[tmp_axis] / _num_splits) : _size_splits[idx];
        if (split_size == -1)
        {
            split_size = input_descriptor.shape[tmp_axis];
            for (unsigned int i = 0; i < _size_splits.size() - 1; ++i)
                split_size -= _size_splits[i];
        }
        output_descriptor.shape.set(tmp_axis, split_size);

        return output_descriptor;
    }

    BIStatus SplitLayerNode::validate() const
    {
        const BITensor *src = input(0);
        BI_COMPUTE_RETURN_ERROR_ON(src == nullptr);
        int num_dimension = static_cast<int32_t>(src->desc().shape.num_dimensions());
        BI_COMPUTE_RETURN_ERROR_ON(_axis < (-num_dimension) || _axis >= num_dimension);

        // Handle negative axis, negative index is used to specify axis from the end (e.g. -1 for the last axis).
        int tmp_axis = wrap_around(_axis, num_dimension);

        if (_size_splits.empty())
        {
            BI_COMPUTE_RETURN_ERROR_ON_MSG(src->desc().shape[tmp_axis] % _num_splits, "Split should be exact");
        }

        return BIStatus{};
    }

    BINodeType SplitLayerNode::type() const
    {
        return BINodeType::SplitLayer;
    }

    void SplitLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
