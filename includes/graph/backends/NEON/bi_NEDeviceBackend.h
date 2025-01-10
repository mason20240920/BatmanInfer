//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_NEDEVICEBACKEND_H
#define BATMANINFER_GRAPH_BI_NEDEVICEBACKEND_H

#include "graph/bi_ideviceBackend.h"
#include "runtime/bi_allocator.hpp"

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** CPU device backend */
    class BINEDeviceBackend final : public BIIDeviceBackend
    {
    public:
        BINEDeviceBackend();

        // Inherited overridden methods
        void                             initialize_backend() override;
        void                             setup_backend_context(BIGraphContext &ctx) override;
        void                             release_backend_context(BIGraphContext &ctx) override;
        bool                             is_backend_supported() override;
        BIIAllocator                    *backend_allocator() override;
        std::unique_ptr<BIITensorHandle> create_tensor(const BITensor &tensor) override;
        std::unique_ptr<BIITensorHandle>
        create_subtensor(BIITensorHandle *parent, BITensorShape shape, BICoordinates coords, bool extend_parent) override;
        std::unique_ptr<BatmanInfer::BIIFunction>       configure_node(BIINode &node, BIGraphContext &ctx) override;
        BIStatus                                        validate_node(BIINode &node) override;
        std::shared_ptr<BatmanInfer::BIIMemoryManager>  create_memory_manager(BIMemoryManagerAffinity affinity) override;
        std::shared_ptr<BatmanInfer::BIIWeightsManager> create_weights_manager() override;
        void                                            sync() override;

    private:
        BIAllocator _allocator; /**< Backend allocator */
    };

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_NEDEVICEBACKEND_H
