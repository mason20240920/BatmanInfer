//
// Created by holynova on 2025/1/2.
//

#ifndef BATMANINFER_BI_INODE_H
#define BATMANINFER_BI_INODE_H

#include "data/core/bi_error.h"
#include "graph/bi_layerDescriptors.h"
#include "graph/bi_tensorDescriptor.h"
#include "graph/bi_types.h"

#include <set>

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;
    class BIEdge;
    class BIINodeVisitor;
    class BITensor;

    /** Node interface */
    class BIINode
    {
    public:
        /** Constructor */
        BIINode();
        /** Destructor **/
        virtual ~BIINode() = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIINode(const BIINode &) = delete;
        /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
        BIINode &operator=(const BIINode &) = delete;
        /** Allow instances of this class to be moved */
        BIINode(BIINode &&) = default;
        /** Allow instances of this class to be move assigned */
        BIINode &operator=(BIINode &&) = default;
        /** Validate node
         *
         * @return Status containing any errors
         */
        virtual BIStatus validate() const;
        /** Returns node's type
         *
         * @return Node's type
         */
        virtual BINodeType type() const = 0;
        /** Accepts a node visitor
         *
         * @param[in] v Visitor to accept
         */
        virtual void accept(BIINodeVisitor &v) = 0;
        /** Forwards descriptor information to outputs if possible
         *
         * @return True if descriptor information could be forwarded otherwise false
         */
        virtual bool forward_descriptors() = 0;
        /** Calculates output configuration
         *
         * @param[in] idx Output index to configure
         *
         * @return Output descriptor configuration
         */
        virtual BITensorDescriptor configure_output(size_t idx) const = 0;
        /** Returns node's name
         *
         * @return Node name
         */
        std::string name() const;
        /** Returns node's ID
         *
         * @return Node's ID
         */
        NodeID id() const;
        /** Returns node's Graph
         *
         * @return Node's graph
         */
        const BIGraph *graph() const;
        /** Returns node's Graph
         *
         * @return Node's graph
         */
        BIGraph *graph();
        /** Sets the graph that this node is registered to
         *
         * @param[in] g Back reference to graph
         */
        void set_graph(BIGraph *g);
        /** Sets the node id
         *
         * @param[in] id Node id
         */
        void set_id(NodeID id);
        /** Sets common node parameters
         *
         * @param[in] common_params Common node parameters to set
         */
        void set_common_node_parameters(BINodeParams common_params);
        /** Sets target preference
         *
         * @note This is not the target that the graph executor might choose, its just an indication
         *
         * @param[in] target Target preference
         */
        void set_requested_target(BITarget target);
        /** Sets the final execution target
         *
         * @note GraphManager might change this target
         *
         * @param[in] target Final execution target
         */
        void set_assigned_target(BITarget target);
        /** Sets the output tensor of at a given index
         *
         * @note All edges will get updated
         *
         * @param[in] tid Tensor ID
         * @param[in] idx Output index
         */
        void set_output_tensor(TensorID tid, size_t idx);
        /** Returns inputs of the node
         *
         * @return Inputs of the node
         */
        const std::vector<TensorID> &inputs() const;
        /** Returns outputs of the node
         *
         * @return Outputs of the node
         */
        const std::vector<TensorID> &outputs() const;
        /** Returns input edge set
         *
         * @return Set of input edges
         */
        const std::vector<EdgeID> &input_edges() const;
        /** Returns output edge set
         *
         * @return Set of output edges
         */
        const std::set<EdgeID> &output_edges() const;
        /** Returns the tensor ID of a given input of the node
         *
         * @note Precondition : idx should be a valid input index
         *
         * @param[in] idx Index of the node input
         *
         * @return TensorID of the requested input
         */
        TensorID input_id(size_t idx) const;
        /** Returns the tensor ID of a given output of the node
         *
         * @note Precondition : idx should be a valid output index
         *
         * @param[in] idx Index of the node output
         *
         * @return TensorID of the requested output
         */
        TensorID output_id(size_t idx) const;
        /** Returns the tensor of a given input of the node
         *
         * @note Precondition : idx should be a valid input index
         *
         * @param[in] idx Index of the node input
         *
         * @return Tensor of the requested input
         */
        BITensor *input(size_t idx) const;
        /** Returns the tensor of a given output of the node
         *
         * @note Precondition : idx should be a valid output index
         *
         * @param[in] idx Index of the node output
         *
         * @return Tensor of the requested output
         */
        BITensor *output(size_t idx) const;
        /** Returns the edge ID of a given input of the node
         *
         * @note Precondition : idx should be a valid input index
         *
         * @param[in] idx Index of the node input
         *
         * @return EdgeID of the requested input
         */
        EdgeID input_edge_id(size_t idx) const;
        /** Returns the edge of a given input of the node
         *
         * @note Precondition : idx should be a valid input index
         *
         * @param[in] idx Index of the node input
         *
         * @return Edge of the requested input
         */
        BIEdge *input_edge(size_t idx) const;
        /** Returns number of inputs of the node
         *
         * @return Number of inputs
         */
        size_t num_inputs() const;
        /** Returns number of outputs of the node
         *
         * @return Number of outputs
         */
        size_t num_outputs() const;
        /** Returns common node parameters
         *
         * @return Common node parameters
         */
        BINodeParams common_node_params() const;
        /** Returns requested target for this node
         *
         * @return Requested execution target
         */
        BITarget requested_target() const;
        /** Returns assigned target for this node
         *
         * @return Assigned target of this node
         */
        BITarget assigned_target() const;

    protected:
        friend class Graph;

    protected:
        BIGraph              *_graph;           /**< Backward reference to graph owning the node */
        NodeID                _id;              /**< Node ID */
        BINodeParams          _common_params;   /**< Node common params */
        std::vector<TensorID> _outputs;         /**< Output of the node */
        std::vector<EdgeID>   _input_edges;     /**< Inputs edge set */
        std::set<EdgeID>      _output_edges;    /**< Output edge set */
        BITarget              _assigned_target; /**< Assigned target by the Graph executor */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_BI_INODE_H
