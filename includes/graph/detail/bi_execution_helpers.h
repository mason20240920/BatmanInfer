//
// Created by holynova on 2025/1/24.
//

#pragma once

#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;
    class BIGraphContext;
    struct BIExecutionWorkload;
    class BITensor;
    class BIINode;

    namespace detail {

        /** Validates all nodes
         *
         * @param[in] g Graph to validate
         */
        void validate_all_nodes(BIGraph &g);

        /** Configures all nodes of a graph
         *
         * @param[in] g Graph to configure
         */
        void configure_all_tensors(BIGraph &g);

        /** Allocates all input tensors of a node.
         *
         * @param[in] node Node to allocate the input tensor of
         */
        void allocate_all_input_tensors(BIINode &node);

        /** Allocates all output tensors of a node.
         *
         * @param[in] node Node to allocate the output tensor of
         */
        void allocate_all_output_tensors(BIINode &node);

        /** Allocates const tensor of a given graph
         *
         * @param[in] g Graph to allocate the tensors
         */
        void allocate_const_tensors(BIGraph &g);

        /** Allocates all tensors of a graph
         *
         * @param[in] g Graph to allocate the tensors
         */
        void allocate_all_tensors(BIGraph &g);

        /** Configures all nodes of graph
         *
         * @param[in, out] g          Graph to configure the nodes
         * @param[in]      ctx        Graph context to use
         * @param[in]      node_order The order to configure the nodes
         *
         * @return The execution workload
         */
        BIExecutionWorkload configure_all_nodes(BIGraph &g, BIGraphContext &ctx, const std::vector<NodeID> &node_order);

        /** Release the memory of all unused const nodes
         *
         * @param[in] g Graph to release the memory from
         */
        void release_unused_tensors(BIGraph &g);

        /** Calls accessor of a given tensor
         *
         * @param[in] tensor The tensor of which the accessor should be called
         */
        void call_tensor_accessor(BITensor *tensor);

        /** Call all const node accessors
         *
         * @param[in] g Graph containing the const nodes
         */
        void call_all_const_node_accessors(BIGraph &g);

        /** Call all input node accessors
         *
         * @param[in] workload Workload to execute
         *
         * @return  True if all the accesses were valid
         */
        bool call_all_input_node_accessors(BIExecutionWorkload &workload);

        /** Call all output node accessors
         *
         * @param[in] workload Workload to execute
         *
         * @return  True if all the accessors expect more data
         */
        bool call_all_output_node_accessors(BIExecutionWorkload &workload);

        /** Prepares all tasks for execution
         *
         * @param[in] workload Workload to prepare
         */
        void prepare_all_tasks(BIExecutionWorkload &workload);

        /** Executes all tasks of a workload
         *
         * @param[in] workload Workload to execute
         */
        void call_all_tasks(BIExecutionWorkload &workload);

    } // namespace detail

} // namespace graph

} // namespace BatmanInfer
