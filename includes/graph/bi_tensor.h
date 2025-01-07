//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_TENSOR_H
#define BATMANINFER_GRAPH_BI_TENSOR_H

#include "graph/bi_itensorAccessor.h"
#include "graph/bi_itensorHandle.h"
#include "graph/bi_tensorDescriptor.h"
#include "graph/bi_types.h"

#include <memory>
#include <set>

namespace BatmanInfer {

namespace graph {

    /** Tensor object **/
    class BITensor final
    {
    public:
        /** Default constructor
         *
         * @param[in] id   Tensor ID
         * @param[in] desc Tensor information
         */
        BITensor(TensorID id, BITensorDescriptor desc);
        /** Tensor ID accessor
         *
         * @return Tensor ID
         */
        TensorID id() const;
        /** TensorInfo metadata accessor
         *
         * @return Tensor descriptor metadata
         */
        BITensorDescriptor &desc();
        /** TensorInfo metadata accessor
         *
         * @return Tensor descriptor metadata
         */
        const BITensorDescriptor &desc() const;
        /** Sets the backend tensor
         *
         * @param[in] backend_tensor Backend tensor to set
         */
        void set_handle(std::unique_ptr<BIITensorHandle> backend_tensor);
        /** Backend tensor handle accessor
         *
         * @return Backend tensor handle
         */
        BIITensorHandle *handle();
        /** Sets the backend tensor accessor
         *
         * @param[in] accessor Accessor to set
         */
        void set_accessor(std::unique_ptr<BIITensorAccessor> accessor);
        /** Backend tensor accessor
         *
         * @return Backend tensor accessor
         */
        BIITensorAccessor *accessor();
        /** Extracts accessor from the tensor
         *
         * @warning Accessor gets unbound from the tensor
         *
         * @return The accessor of the tensor
         */
        std::unique_ptr<BIITensorAccessor> extract_accessor();
        /** Calls accessor on tensor
         *
         * @return True if the accessor was called else false
         */
        bool call_accessor();
        /** Binds the tensor with an edge
         *
         * @param[in] eid Edge ID that is bound to the tensor
         */
        void bind_edge(EdgeID eid);
        /** Unbinds an edge from a tensor
         *
         * @param[in] eid Edge to unbind
         */
        void unbind_edge(EdgeID eid);
        /** Accessor the edges that are bound with the tensor
         *
         * @return Bound edges
         */
        std::set<EdgeID> bound_edges() const;

    private:
        TensorID                           _id;          /**< Tensor id */
        BITensorDescriptor                 _desc;        /**< Tensor metadata */
        std::unique_ptr<BIITensorHandle>   _handle;      /**< Tensor Handle */
        std::unique_ptr<BIITensorAccessor> _accessor;    /**< Tensor Accessor */
        std::set<EdgeID>                   _bound_edges; /**< Edges bound to this tensor */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_TENSOR_H
