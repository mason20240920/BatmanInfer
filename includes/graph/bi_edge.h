//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_EDGE_H
#define BATMANINFER_GRAPH_BI_EDGE_H

#include "graph/bi_inode.h"
#include "graph/bi_tensor.h"
#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;

    /** Graph Edge */
    class BIEdge final
    {
    public:
        /** Default Constructor
         *
         * @param[in] id           Edge id
         * @param[in] producer     Producer node id
         * @param[in] producer_idx Producer node output index
         * @param[in] consumer     Consumer node id
         * @param[in] consumer_idx Consumer node input index
         * @param[in] tensor       Tensor associated with the edge
         */
        BIEdge(EdgeID       id,
               BIINode     *producer,
               unsigned int producer_idx,
               BIINode     *consumer,
               unsigned int consumer_idx,
               BITensor    *tensor)
            : _id(id),
              _producer(producer),
              _consumer(consumer),
              _producer_idx(producer_idx),
              _consumer_idx(consumer_idx),
              _tensor(tensor)

        {
        }
        /** Returns edge id
         *
         * @return Edge id
         */
        EdgeID id() const
        {
            return _id;
        }
        /** Returns producer node id
         *
         * @return Producer node id
         */
        NodeID producer_id() const
        {
            return (_producer == nullptr) ? EmptyNodeID : _producer->id();
        }
        /** Returns sink node id
         *
         * @return Sink node id
         */
        NodeID consumer_id() const
        {
            return (_consumer == nullptr) ? EmptyNodeID : _consumer->id();
        }
        /** Returns producer node
         *
         * @return Producer node
         */
        BIINode *producer() const
        {
            return _producer;
        }
        /** Returns consumer node
         *
         * @return Consumer node
         */
        BIINode *consumer() const
        {
            return _consumer;
        }
        /** Returns the index of the output that produces the result in the producer node
         *
         * @return Producer node output index
         */
        unsigned int producer_idx() const
        {
            return _producer_idx;
        }
        /** Returns the index of the input that consumes the result in the consumer node
         *
         * @return Consumer node input index
         */
        unsigned int consumer_idx() const
        {
            return _consumer_idx;
        }
        /** Returns the tensor associated with this edge
         *
         * @return Tensor id
         */
        BITensor *tensor() const
        {
            return _tensor;
        }
        /** Returns the tensor id associated with this edge
         *
         * @return Tensor id
         */
        TensorID tensor_id() const
        {
            return (_tensor == nullptr) ? NullTensorID : _tensor->id();
        }
        /** Bind the edge to another tensor
         *
         * @note If tensor is nullptr then nothing happens
         *
         * @param[in] tensor Tensor to bind the edge to
         */
        void update_bound_tensor(BITensor *tensor)
        {
            _tensor = (tensor != nullptr) ? tensor : _tensor;
        }

    private:
        friend class BIGraph;

    private:
        EdgeID       _id;
        BIINode     *_producer;
        BIINode     *_consumer;
        unsigned int _producer_idx;
        unsigned int _consumer_idx;
        BITensor    *_tensor;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_EDGE_H
