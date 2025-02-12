//
// Created by holynova on 2025/1/24.
//

#include "graph/detail/bi_execution_helpers.h"

#include "graph/backends/bi_backendRegistry.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphContext.h"
#include "graph/bi_graphManager.h"
#include "graph/bi_tensor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

namespace detail {

    void validate_all_nodes(BIGraph &g)
    {
        auto &nodes = g.nodes();

        // Create tasks
        for (auto &node : nodes)
        {
            if (node != nullptr)
            {
                BITarget assigned_target = node->assigned_target();
                backends::BIIDeviceBackend &backend =
                    backends::BIBackendRegistry::get().get_backend(assigned_target);
                BIStatus status = backend.validate_node(*node);
                BI_COMPUTE_ERROR_ON_MSG(!bool(status), status.error_description().c_str());
            }
        }
    }

    void configure_all_tensors(BIGraph &g)
    {
        auto &tensors = g.tensors();

        for (auto &tensor : tensors)
        {
            if (tensor && tensor->handle() == nullptr)
            {
                BITarget                         target  = tensor->desc().target;
                backends::BIIDeviceBackend      &backend = backends::BIBackendRegistry::get().get_backend(target);
                std::unique_ptr<BIITensorHandle> handle  = backend.create_tensor(*tensor);
                BI_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
                tensor->set_handle(std::move(handle));
            }
        }
    }

    void allocate_all_input_tensors(BIINode &node)
    {
        for (unsigned int i = 0; i < node.num_inputs(); ++i)
        {
            BITensor *tensor = node.input(i);
            if (tensor != nullptr && !tensor->bound_edges().empty())
            {
                BI_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
                tensor->handle()->allocate();
            }
        }
    }

    void allocate_all_output_tensors(BIINode &node)
    {
        for (unsigned int i = 0; i < node.num_outputs(); ++i)
        {
            BITensor *tensor = node.output(i);
            if (tensor != nullptr && !tensor->bound_edges().empty())
            {
                BI_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
                tensor->handle()->allocate();
            }
        }
    }

    void allocate_const_tensors(BIGraph &g)
    {
        for (auto &node : g.nodes())
        {
            if (node != nullptr)
            {
                switch (node->type())
                {
                case BINodeType::Const:
                case BINodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case BINodeType::Output:
                    allocate_all_input_tensors(*node);
                default:
                    break;
                }
            }
        }
    }

    void allocate_all_tensors(BIGraph &g)
    {
        auto &tensors = g.tensors();

        for (auto &tensor : tensors)
        {
            if (tensor && !tensor->bound_edges().empty() && tensor->handle() != nullptr &&
                tensor->handle()->tensor().info()->is_resizable() && tensor->handle()->tensor().is_used())
            {
                tensor->handle()->allocate();
            }
        }
    }

    BIExecutionWorkload configure_all_nodes(BIGraph &g, BIGraphContext &ctx, const std::vector<NodeID> &node_order)
    {
        BIExecutionWorkload workload;
        workload.graph = &g;
        workload.ctx   = &ctx;

        // Reserve memory for tasks
        workload.tasks.reserve(node_order.size());

        // Create tasks
        for (auto &node_id : node_order)
        {
            auto node = g.node(node_id);
            if (node != nullptr)
            {
                BITarget                     assigned_target = node->assigned_target();
                backends::BIIDeviceBackend  &backend         = backends::BIBackendRegistry::get().get_backend(assigned_target);
                std::unique_ptr<BIIFunction> func            = backend.configure_node(*node, ctx);
                if (func != nullptr || is_utility_node(node))
                {
                    workload.tasks.emplace_back(BIExecutionTask(std::move(func), node));
                }
            }
        }

        // Add inputs and outputs
        for (auto &node : g.nodes())
        {
            if (node != nullptr && node->type() == BINodeType::Input)
            {
                workload.inputs.push_back(node->output(0));
            }

            if (node != nullptr && node->type() == BINodeType::Output)
            {
                workload.outputs.push_back(node->input(0));
                continue;
            }
        }

        return workload;
    }

    void release_unused_tensors(BIGraph &g)
    {
        for (auto &tensor : g.tensors())
        {
            if (tensor != nullptr && tensor->handle() != nullptr)
            {
                tensor->handle()->release_if_unused();
            }
        }
    }

    void call_tensor_accessor(BITensor *tensor)
    {
        BI_COMPUTE_ERROR_ON(!tensor);
        tensor->call_accessor();
    }

    void call_all_const_node_accessors(BIGraph &g)
    {
        auto &nodes = g.nodes();

        for (auto &node : nodes)
        {
            if (node != nullptr && node->type() == BINodeType::Const && node->num_outputs())
            {
                if (!node->output(0)->bound_edges().empty())
                {
                    call_tensor_accessor(node->output(0));
                }
            }
        }
    }

    bool call_all_input_node_accessors(BIExecutionWorkload &workload)
    {
        bool is_valid = true;
        std::for_each(std::begin(workload.inputs), std::end(workload.inputs),
                      [&](BITensor *input_tensor)
                      {
                          bool valid_input = (input_tensor != nullptr) && input_tensor->call_accessor();
                          is_valid         = is_valid && valid_input;
                      });
        return is_valid;
    }

    void prepare_all_tasks(BIExecutionWorkload &workload)
    {
        BI_COMPUTE_ERROR_ON(workload.graph == nullptr);
        for (auto &task : workload.tasks)
        {
            task.prepare();
            release_unused_tensors(*workload.graph);
        }
    }

    void call_all_tasks(BIExecutionWorkload &workload)
    {
        BI_COMPUTE_ERROR_ON(workload.ctx == nullptr);

        // Acquire memory for the transition buffers
        for (auto &mm_ctx : workload.ctx->memory_managers())
        {
            if (mm_ctx.second.cross_group != nullptr)
            {
                mm_ctx.second.cross_group->acquire();
            }
        }

        // Execute tasks
        for (auto &task : workload.tasks)
        {
            task();
        }

        // Release memory for the transition buffers
        for (auto &mm_ctx : workload.ctx->memory_managers())
        {
            if (mm_ctx.second.cross_group != nullptr)
            {
                mm_ctx.second.cross_group->release();
            }
        }
    }

    bool call_all_output_node_accessors(BIExecutionWorkload &workload)
    {
        bool is_valid = true;
        std::for_each(std::begin(workload.outputs), std::end(workload.outputs),
                      [&](BITensor *output_tensor)
                      {
                          bool valid_output = (output_tensor != nullptr) && output_tensor->call_accessor();
                          is_valid          = is_valid && valid_output;
                      });

        sync_backends();

        return is_valid;
    }

} // namespace detail

} // namespace graph

} // namespace BatmanInfer
