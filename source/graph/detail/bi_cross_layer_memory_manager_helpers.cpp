//
// Created by holynova on 2025/1/24.
//

#include "graph/detail/bi_cross_layer_memory_manager_helpers.h"

#include "data/core/bi_i_tensor.hpp"

#include "graph/backends/bi_backendRegistry.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphContext.h"
#include "graph/bi_graphManager.h"
#include "graph/bi_inode.h"
#include "graph/bi_tensor.h"
#include "graph/bi_types.h"
#include "graph/bi_utils.h"
#include "support/bi_cast.hpp"

#include <algorithm>
#include <map>

namespace BatmanInfer {

namespace graph {

namespace detail {

namespace {

    using HandleCountPair     = std::pair<BIITensorHandle *, unsigned int>;
    using HandleCounter       = std::map<HandleCountPair::first_type, HandleCountPair::second_type>;
    using TargetHandleCounter = std::map<BITarget, HandleCounter>;

    /** Holds managed IO tensor handles if a task */
    struct TaskHandles
    {
        /**< Input handles to a task */
        std::vector<std::pair<BIITensorHandle *, BIIMemoryGroup *>> input_handles  = {};
        /**< Output handles of a task */
        std::vector<std::pair<BIITensorHandle *, BIIMemoryGroup *>> output_handles = {};
    };

    /** Returns memory group depending on handle backend type
     *
     * @param[in] ctx    Graph context
     * @param[in] handle Tensor handle
     *
     * @return Memory groupb
     */
    BIIMemoryGroup *get_memory_group_from_handle(BIGraphContext &ctx, BIITensorHandle *handle)
    {
        BI_COMPUTE_ERROR_ON(handle == nullptr);
        return ctx.memory_management_ctx(handle->target())->cross_group.get();
    }

    /** Get handles of const tensors of graph
     *
     * @param[in] g Graph
     *
     * @return Handles of const tensors of graph
     */
    std::set<BIITensorHandle *> get_const_handles(const BIGraph &g)
    {
        std::set<BINodeType> const_node_types = {BINodeType::Input, BINodeType::Output, BINodeType::Const};

        std::set<BIITensorHandle *> const_tensors;

        auto &nodes = g.nodes();
        for (auto &node : nodes)
        {
            // If its a const node:
            if (node != nullptr && const_node_types.find(node->type()) != std::end(const_node_types))
            {
                // Add all its inputs / outputs to the list of constant handles
                for (unsigned int i = 0; i < node->num_inputs(); ++i)
                {
                    if (node->input(i) != nullptr)
                    {
                        const_tensors.insert(node->input(i)->handle()->parent_handle());
                    }
                }
                for (unsigned int i = 0; i < node->num_outputs(); ++i)
                {
                    if (node->output(i) != nullptr)
                    {
                        const_tensors.insert(node->output(i)->handle()->parent_handle());
                    }
                }
            }
        }

        return const_tensors;
    }

    /** Builds a list of all the transition handles (Handles that are used to link two nodes)
     *
     * @param[in] ctx           Graph context
     * @param[in] task          Workload task
     * @param[in] const_tensors Constant tensors
     *
     * @return List of transition handles
     */
    TaskHandles
    get_transition_handles(BIGraphContext &ctx, BIExecutionTask &task, const std::set<BIITensorHandle *> &const_tensors)
    {
        BI_COMPUTE_ERROR_ON(task.node == nullptr || (task.task == nullptr && !is_utility_node(task.node)));
        BIINode &node = *task.node;

        TaskHandles transition_handles;

        // Add input handles
        for (unsigned int i = 0; i < node.input_edges().size(); ++i)
        {
            BIEdge *input_edge = node.input_edge(i);
            // If this input is the output of another node
            if (input_edge != nullptr && input_edge->tensor() != nullptr &&
                const_tensors.find(input_edge->tensor()->handle()->parent_handle()) == std::end(const_tensors))
            {
                // Then add it to the list of transition buffers
                BIITensorHandle *tensor_handle = input_edge->tensor()->handle()->parent_handle();
                BIIMemoryGroup  *mm_group      = get_memory_group_from_handle(ctx, tensor_handle);
                transition_handles.input_handles.emplace_back(std::make_pair(tensor_handle, mm_group));
            }
        }

        // Add output handles
        for (unsigned int i = 0; i < node.num_outputs(); ++i)
        {
            BITensor *output_tensor = node.output(i);
            // If this output is used as an input for another node
            if (output_tensor != nullptr &&
                const_tensors.find(output_tensor->handle()->parent_handle()) == std::end(const_tensors))
            {
                BIITensorHandle *tensor_handle = output_tensor->handle()->parent_handle();
                BIIMemoryGroup  *mm_group      = get_memory_group_from_handle(ctx, tensor_handle);
                transition_handles.output_handles.emplace_back(std::make_pair(tensor_handle, mm_group));
            }
        }

        return transition_handles;
    }

    /** Counts handles refcount for each input handle of each target
     *
     * @param[in]     task           Execution task containing the managed handles
     * @param[in,out] handle_counter Data structure that keeps the handles reference count
     */
    void count_input_handles_per_target(const TaskHandles &task_handles, TargetHandleCounter &handle_counter)
    {
        for (const auto &handle : task_handles.input_handles)
        {
            BIITensorHandle *key          = handle.first;
            HandleCounter &target_counter = handle_counter[key->target()];
            if (target_counter.find(key) == std::end(target_counter))
            {
                target_counter.emplace(std::make_pair(key, 1));
            }
            else
            {
                ++target_counter[key];
            }
        }
    }

    /** Calculates the lifetime of each tensor handle
     *
     * @param[in, out] tasks_handles Tensor handles for each task
     * @param[in]      hc            Data structure that keeps the handles reference count
     */
    void configure_handle_lifetime(std::vector<TaskHandles> &tasks_handles, const HandleCounter &hc)
    {
        // Identify max number of tensors in flight
        HandleCounter tensors_in_flight;

        // Acquires the given handles and sets them as in flight if they aren't already
        auto acquire = [&](std::vector<std::pair<BIITensorHandle *, BIIMemoryGroup *>> &handles)
        {
            for (auto &handle : handles)
            {
                BIITensorHandle *parent_handle = handle.first;
                BI_COMPUTE_ERROR_ON(parent_handle == nullptr);
                // If the tensor is not already in flight:
                if (tensors_in_flight.find(parent_handle) == std::end(tensors_in_flight))
                {
                    BI_COMPUTE_ERROR_ON(hc.find(parent_handle) == std::end(hc));
                    // Then add it to the list of in flight tensors
                    tensors_in_flight.insert(std::make_pair(parent_handle, hc.at(parent_handle)));
                    // Start of allocation's lifetime
                    parent_handle->manage(handle.second);
                }
            }
        };

        for (auto &task_handle : tasks_handles)
        {
            // Marking all the input and output tensors of the task as in flight
            acquire(task_handle.input_handles);
            acquire(task_handle.output_handles);

            // Releasing the input tensors
            for (auto &input_handle : task_handle.input_handles)
            {
                BIITensorHandle *ihandle = input_handle.first;
                BI_COMPUTE_ERROR_ON(ihandle == nullptr);
                BI_COMPUTE_ERROR_ON(tensors_in_flight.find(ihandle) == std::end(tensors_in_flight));
                --tensors_in_flight[ihandle];
                if (tensors_in_flight[ihandle] <= 0)
                {
                    // Remove tensor for tensors in flight
                    tensors_in_flight.erase(ihandle);
                    // End of allocation's lifetime
                    ihandle->allocate();
                }
            }
        }
    }

} // namespace

void configure_transition_manager(BIGraph &g, BIGraphContext &ctx, BIExecutionWorkload &workload)
{
    // Get const tensors (un-managed)
    std::set<BIITensorHandle *> const_tensors = get_const_handles(g);

    std::vector<TaskHandles> tasks_handles;
    TargetHandleCounter      target_handle_count;

    // Count handles
    for (auto &task : workload.tasks)
    {
        // Populates IO handles
        tasks_handles.push_back(get_transition_handles(ctx, task, const_tensors));

        // Count handles
        count_input_handles_per_target(tasks_handles.back(), target_handle_count);
    }

    // Setup memory managers
    for (auto &hc : target_handle_count)
    {
        BIMemoryManagerContext *mm_ctx = ctx.memory_management_ctx(hc.first);
        if (mm_ctx != nullptr)
        {
            if (mm_ctx->cross_mm != nullptr && mm_ctx->cross_group != nullptr)
            {
                // Manage and allocate tensors
                configure_handle_lifetime(tasks_handles, hc.second);
            }
        }
    }
}

} // namespace detail

} // namespace graph

} // namespace BatmanInfer
