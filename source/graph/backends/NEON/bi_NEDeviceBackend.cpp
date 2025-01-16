//
// Created by holynova on 2025/1/10.
//

#include "graph/backends/NEON/bi_NEDeviceBackend.h"

#include "data/core/bi_tensor_info.hpp"
#include "graph/backends/bi_backendRegistrar.h"
#include "graph/backends/NEON/bi_NEFunctionFactory.h"
#include "graph/backends/NEON/bi_NENodeValidator.h"
#include "graph/backends/NEON/bi_NETensorHandle.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphContext.h"
#include "graph/bi_inode.h"
#include "graph/bi_logger.h"
#include "graph/bi_tensor.h"
#include "runtime/bi_allocator.hpp"
#include "runtime/bi_blob_lifetime_manager.hpp"
#include "runtime/bi_i_weights_manager.hpp"
#include "runtime/bi_memory_group.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/bi_offset_lifetime_manager.h"
#include "runtime/bi_pool_manager.hpp"
#include "runtime/bi_scheduler.hpp"

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** Register CPU backend */
    static detail::BackendRegistrar<BINEDeviceBackend> NEDeviceBackend_registrar(BITarget::NEON);

    BINEDeviceBackend::BINEDeviceBackend() : _allocator()
    {
    }

    void BINEDeviceBackend::initialize_backend()
    {
        //Nothing to do
    }

    void BINEDeviceBackend::release_backend_context(BIGraphContext &ctx)
    {
        //Nothing to do
        BI_COMPUTE_UNUSED(ctx);
    }

    void BINEDeviceBackend::setup_backend_context(BIGraphContext &ctx)
    {
        // Set number of threads
        if (ctx.config().num_threads >= 0)
        {
            BIScheduler::get().set_num_threads(ctx.config().num_threads);
        }

        // Create function level memory manager
        if (ctx.memory_management_ctx(BITarget::NEON) == nullptr)
        {
            BIMemoryManagerContext mm_ctx;
            mm_ctx.target      = BITarget::NEON;
            mm_ctx.intra_mm    = create_memory_manager(BIMemoryManagerAffinity::Offset);
            mm_ctx.cross_mm    = create_memory_manager(BIMemoryManagerAffinity::Offset);
            mm_ctx.cross_group = std::make_shared<BIMemoryGroup>(mm_ctx.cross_mm);
            mm_ctx.allocator   = &_allocator;

            ctx.insert_memory_management_ctx(std::move(mm_ctx));
        }

        // Create function level weights manager
        if (ctx.weights_management_ctx(BITarget::NEON) == nullptr)
        {
            BIWeightsManagerContext wm_ctx;
            wm_ctx.target = BITarget::NEON;
            wm_ctx.wm     = create_weights_manager();

            ctx.insert_weights_management_ctx(std::move(wm_ctx));
        }
    }

    bool BINEDeviceBackend::is_backend_supported()
    {
        return true;
    }

    BIIAllocator *BINEDeviceBackend::backend_allocator()
    {
        return &_allocator;
    }

    std::unique_ptr<BIITensorHandle> BINEDeviceBackend::create_tensor(const BITensor &tensor)
    {
        // Get tensor descriptor
        const BITensorDescriptor &tensor_desc = tensor.desc();
        BI_COMPUTE_ERROR_ON(tensor_desc.target != BITarget::NEON);

        // Create backend tensor handle
        BITensorInfo info(tensor_desc.shape, 1, tensor_desc.data_type, tensor_desc.quant_info);
        // info.set_data_layout(tensor_desc.layout);

        return std::make_unique<BINETensorHandle>(info);
    }

    std::unique_ptr<BatmanInfer::BIIFunction> BINEDeviceBackend::configure_node(BIINode &node, BIGraphContext &ctx)
    {
        BI_COMPUTE_LOG_GRAPH_VERBOSE("Configuring CPU node with ID : " << node.id() << std::endl);
        BI_COMPUTE_ERROR_ON(node.assigned_target() != Target::NEON);

        // Configure node
        return BINEFunctionFactory::create(&node, ctx);
    }

    BatmanInfer::BIStatus BINEDeviceBackend::validate_node(BIINode &node)
    {
        BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating CPU node with ID : " << node.id() << std::endl);
        BI_COMPUTE_ERROR_ON(node.assigned_target() != BITarget::NEON);

        return BINENodeValidator::validate(&node);
    }

    std::shared_ptr<BatmanInfer::BIIMemoryManager> BINEDeviceBackend::create_memory_manager(BIMemoryManagerAffinity affinity)
    {
        std::shared_ptr<BIILifetimeManager> lifetime_mgr = nullptr;
        if (affinity == BIMemoryManagerAffinity::Buffer)
        {
            lifetime_mgr = std::make_shared<BIBlobLifetimeManager>();
        }
        else
        {
            lifetime_mgr = std::make_shared<BIOffsetLifetimeManager>();
        }
        auto pool_mgr = std::make_shared<BIPoolManager>();
        auto mm       = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

        return mm;
    }

    std::shared_ptr<BatmanInfer::BIIWeightsManager> BINEDeviceBackend::create_weights_manager()
    {
        auto weights_mgr = std::make_shared<BIIWeightsManager>();
        return weights_mgr;
    }

    void BINEDeviceBackend::sync()
    {
    }

} // namespace backends

} // namespace graph

} // namespace BatmanInfer
