//
// Created by holynova on 2025/1/7.
//

#include "graph/bi_tensor.h"

namespace BatmanInfer {

namespace graph {

    BITensor::BITensor(TensorID id, BITensorDescriptor desc)
    : _id(id), _desc(std::move(desc)), _handle(nullptr), _accessor(nullptr), _bound_edges()
    {
    }

    TensorID BITensor::id() const
    {
        return _id;
    }

    BITensorDescriptor &BITensor::desc()
    {
        return _desc;
    }

    const BITensorDescriptor &BITensor::desc() const
    {
        return _desc;
    }

    void BITensor::set_handle(std::unique_ptr<BIITensorHandle> backend_tensor)
    {
        _handle = std::move(backend_tensor);
    }

    BIITensorHandle *BITensor::handle()
    {
        return _handle.get();
    }

    void BITensor::set_accessor(std::unique_ptr<BIITensorAccessor> accessor)
    {
        _accessor = std::move(accessor);
    }

    BIITensorAccessor *BITensor::accessor()
    {
        return _accessor.get();
    }

    std::unique_ptr<BIITensorAccessor> BITensor::extract_accessor()
    {
        return std::move(_accessor);
    }

    bool BITensor::call_accessor()
    {
        // Early exit guard
        if (!_accessor || !_handle)
        {
            return false;
        }

        const bool access_data = _accessor->access_tensor_data();

        if (access_data)
        {
            // Map tensor
            _handle->map(true);

            // Return in case of null backend buffer
            if (_handle->tensor().buffer() == nullptr)
            {
                return false;
            }
        }

        // Call accessor
        bool retval = _accessor->access_tensor(_handle->tensor());

        if (access_data)
        {
            // Unmap tensor
            _handle->unmap();
        }

        return retval;
    }

    void BITensor::bind_edge(EdgeID eid)
    {
        _bound_edges.insert(eid);
    }

    void BITensor::unbind_edge(EdgeID eid)
    {
        _bound_edges.erase(eid);
    }

    std::set<EdgeID> BITensor::bound_edges() const
    {
        return _bound_edges;
    }

} // namespace graph

} // namespace BatmanInfer
