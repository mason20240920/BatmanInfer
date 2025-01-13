//
// Created by holynova on 2025/1/13.
//

#include "graph/bi_utils.h"

#include "graph/backends/bi_backendRegistry.h"
#include "graph/bi_graphContext.h"
// #include "graph/mutators/bi_graphMutators.h"

namespace BatmanInfer {

namespace graph {

    bool is_target_supported(BITarget target)
    {
        return backends::BIBackendRegistry::get().contains(target) &&
               backends::BIBackendRegistry::get().find_backend(target)->is_backend_supported();
    }

    BITarget get_default_target()
    {
        if (is_target_supported(BITarget::NEON))
        {
            return BITarget::NEON;
        }
        if (is_target_supported(BITarget::CL))
        {
            return BITarget::CL;
        }
        BI_COMPUTE_ERROR("No backend exists!");
    }

    void force_target_to_graph(BIGraph &g, BITarget target)
    {
        auto &nodes = g.nodes();
        for (auto &node : nodes)
        {
            if (node)
            {
                node->set_assigned_target(target);
            }
        }

        auto &tensors = g.tensors();
        for (auto &tensor : tensors)
        {
            if (tensor)
            {
                tensor->desc().target = target;
            }
        }
    }

    /*
    BIPassManager create_default_pass_manager(BITarget target, const BIGraphConfig &cfg)
    {
        BI_COMPUTE_UNUSED(target);
        BIPassManager pm;

        // Passes that mutate graph IR
        if (cfg.use_synthetic_type)
        {
            switch (cfg.synthetic_type)
            {
            case BIDataType::QASYMM8:
            case BIDataType::QASYMM8_SIGNED:
                {
                    pm.append(std::make_unique<SyntheticDataTypeMutator>(cfg.synthetic_type));
                    break;
                }
            default:
                {
                    BI_COMPUTE_ERROR("Unsupported DataType for SyntheticDataTypeMutator");
                    break;
                }
            }
        }
        pm.append(std::make_unique<NodeFusionMutator>());
        pm.append(std::make_unique<GroupedConvolutionMutator>());
        pm.append(std::make_unique<InPlaceOperationMutator>());

        // Passes that mutate backend information
        pm.append(std::make_unique<DepthConcatSubTensorMutator>());
        pm.append(std::make_unique<SplitLayerSubTensorMutator>());
        pm.append(std::make_unique<NodeExecutionMethodMutator>());

        return pm;
    }
     */

    void release_default_graph_context(BIGraphContext &ctx)
    {
        for (const auto &backend : backends::BIBackendRegistry::get().backends())
        {
            if (backend.second->is_backend_supported())
            {
                backend.second->release_backend_context(ctx);
            }
        }
    }

    void sync_backends()
    {
        for (const auto &backend : backends::BIBackendRegistry::get().backends())
        {
            if (backend.second->backend_allocator())
            {
                backend.second->sync();
            }
        }
    }

    void setup_requested_backend_context(BIGraphContext &ctx, BITarget target)
    {
        if (backends::BIBackendRegistry::get().contains(target))
        {
            const auto &backend = backends::BIBackendRegistry::get().find_backend(target);
            if (backend->is_backend_supported())
            {
                backend->setup_backend_context(ctx);
            }
        }
    }

    size_t get_dimension_size(const BITensorDescriptor &descriptor, const BIDataLayoutDimension data_layout_dimension)
    {
        BI_COMPUTE_ERROR_ON_MSG(descriptor.layout == BIDataLayout::UNKNOWN,
                                "Cannot retrieve the dimension index for an unknown layout!");
        return descriptor.shape[get_dimension_idx(descriptor.layout, data_layout_dimension)];
    }

    size_t get_dimension_idx(BIDataLayout data_layout, const BIDataLayoutDimension data_layout_dimension)
    {
        BI_COMPUTE_ERROR_ON_MSG(data_layout == BIDataLayout::UNKNOWN,
                                "Cannot retrieve the dimension index for an unknown layout!");

        /* Return the index based on the data layout
         * [N C H W]
         * [3 2 1 0]
         * [N H W C]
         */
        switch (data_layout_dimension)
        {
        case BIDataLayoutDimension::CHANNEL:
            return (data_layout == BIDataLayout::NCHW) ? 2 : 0;
            break;
        case BIDataLayoutDimension::HEIGHT:
            return (data_layout == BIDataLayout::NCHW) ? 1 : 2;
            break;
        case BIDataLayoutDimension::WIDTH:
            return (data_layout == BIDataLayout::NCHW) ? 0 : 1;
            break;
        case BIDataLayoutDimension::BATCHES:
            return 3;
            break;
        default:
            break;
        }
        BI_COMPUTE_ERROR("Data layout index not supported!");
    }

    std::vector<BINodeIdxPair> get_driving_nodes(const BIINode &node)
    {
        std::vector<BINodeIdxPair> driving_nodes;

        const BIGraph *g = node.graph();
        BI_COMPUTE_ERROR_ON(g == nullptr);

        for (auto &output_edge_id : node.output_edges())
        {
            auto output_edge = g->edge(output_edge_id);
            if (output_edge != nullptr)
            {
                BI_COMPUTE_ERROR_ON(output_edge->consumer() == nullptr);
                driving_nodes.push_back({output_edge->consumer_id(), output_edge->consumer_idx()});
            }
        }

        return driving_nodes;
    }

    std::vector<BINodeIdxPair> get_driver_nodes(const BIINode &node)
    {
        std::vector<BINodeIdxPair> driver_nodes;

        const BIGraph *g = node.graph();
        BI_COMPUTE_ERROR_ON(g == nullptr);

        for (auto &input_edge_id : node.input_edges())
        {
            auto input_edge = g->edge(input_edge_id);
            if (input_edge != nullptr)
            {
                BI_COMPUTE_ERROR_ON(input_edge->producer() == nullptr);
                driver_nodes.push_back({input_edge->producer_id(), input_edge->producer_idx()});
            }
        }

        return driver_nodes;
    }

    void configure_tensor(BITensor *tensor)
    {
        if (tensor != nullptr && tensor->handle() == nullptr)
        {
            BITarget                         target  = tensor->desc().target;
            backends::BIIDeviceBackend      &backend = backends::BIBackendRegistry::get().get_backend(target);
            std::unique_ptr<BIITensorHandle> handle  = backend.create_tensor(*tensor);
            BI_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }

} // namespace graph

} // namespace BatmanInfer
