//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_IDEVICEBACKEND_H
#define BATMANINFER_GRAPH_BI_IDEVICEBACKEND_H

#include "graph/bi_itensorHandle.h"
#include "graph/bi_types.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_i_weights_manager.hpp"

namespace BatmanInfer {

namespace graph {

// Forward declarations
class BIGraph;
class BIGraphContext;
class BITensor;
class BIINode;

namespace backends {

    /** Device backend interface */
    class BIIDeviceBackend
    {
    public:
        /** Virtual Destructor */
        virtual ~BIIDeviceBackend() = default;
        /** Initializes the backend */
        virtual void initialize_backend() = 0;
        /** Setups the given graph context
         *
         * @param[in,out] ctx Graph context
         */
        virtual void setup_backend_context(BIGraphContext &ctx) = 0;
        /** Release the backend specific resources associated to a given graph context
         *
         * @param[in,out] ctx Graph context
         */
        virtual void release_backend_context(BIGraphContext &ctx) = 0;
        /** Checks if an instantiated backend is actually supported
         *
         * @return True if the backend is supported else false
         */
        virtual bool is_backend_supported() = 0;
        /** Gets a backend memory allocator
         *
         * @return Backend memory allocator
         */
        virtual BIIAllocator *backend_allocator() = 0;
        /** Create a backend Tensor
         *
         * @param[in] tensor The tensor we want to create a backend tensor for
         *
         * @return Backend tensor handle
         */
        virtual std::unique_ptr<BIITensorHandle> create_tensor(const BITensor &tensor) = 0;
        /** Create a backend Sub-Tensor
         *
         * @param[in] parent        Parent sub-tensor handle
         * @param[in] shape         Shape of the sub-tensor
         * @param[in] coords        Starting coordinates of the sub-tensor
         * @param[in] extend_parent Extends parent shape if true
         *
         * @return Backend sub-tensor handle
         */
        virtual std::unique_ptr<BIITensorHandle>
        create_subtensor(BIITensorHandle *parent, BITensorShape shape, BICoordinates coords, bool extend_parent) = 0;
        /** Configure a backend Node
         *
         * @note This creates an appropriate configured backend function for the given node
         *
         * @param[in] node The node we want to configure
         * @param[in] ctx  Context to use
         *
         * @return Backend execution function
         */
        virtual std::unique_ptr<BatmanInfer::BIIFunction> configure_node(BIINode &node, BIGraphContext &ctx) = 0;
        /** Validate a node
         *
         * @param[in] node The node we want to validate
         *
         * @return An error status
         */
        virtual BIStatus validate_node(BIINode &node) = 0;
        /** Create a backend memory manager given its affinity
         *
         * @param[in] affinity Memory Manager affinity
         *
         * @return Memory manager
         */
        virtual std::shared_ptr<BatmanInfer::BIIMemoryManager> create_memory_manager(BIMemoryManagerAffinity affinity) = 0;
        /** Create a backend weights manager
         *
         * @return Weights manager
         */
        virtual std::shared_ptr<BatmanInfer::BIIWeightsManager> create_weights_manager() = 0;
        /** Synchronize kernels execution on the backend. On GPU, this results in a blocking call waiting for all kernels to be completed. */
        virtual void sync() = 0;
    };

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_IDEVICEBACKEND_H
