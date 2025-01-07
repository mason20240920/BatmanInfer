//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_GRAPHCONTEXT_H
#define BATMANINFER_GRAPH_BI_GRAPHCONTEXT_H

#include "graph/bi_types.h"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_i_weights_manager.hpp"

namespace BatmanInfer {

namespace graph {

    /** Contains structs required for memory management */
    struct BIMemoryManagerContext
    {
        BITarget                                       target      = {BITarget::UNSPECIFIED}; /**< Target */
        std::shared_ptr<BatmanInfer::BIIMemoryManager> intra_mm    = {nullptr}; /**< Intra-function memory manager */
        std::shared_ptr<BatmanInfer::BIIMemoryManager> cross_mm    = {nullptr}; /**< Cross-function memory manager */
        std::shared_ptr<BatmanInfer::BIIMemoryGroup>   cross_group = {nullptr}; /**< Cross-function memory group */
        BIIAllocator                                  *allocator   = {nullptr}; /**< Backend allocator to use */
    };

    /** Contains structs required for weights management */
    struct BIWeightsManagerContext
    {
        BITarget                                        target = {BITarget::UNSPECIFIED}; /**< Target */
        std::shared_ptr<BatmanInfer::BIIWeightsManager> wm     = {nullptr};               /**< Weights manager */
    };

    /** Graph context **/
    class BIGraphContext final
    {
    public:
        /** Constructor */
        BIGraphContext();
        /** Destructor */
        ~BIGraphContext();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraphContext(const BIGraphContext &) = delete;
        /** Default move constructor */
        BIGraphContext(BIGraphContext &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraphContext &operator=(const BIGraphContext &) = delete;
        /** Default move assignment operator */
        BIGraphContext &operator=(BIGraphContext &&) = default;
        /** Graph configuration accessor
         *
         * @note Every alteration has to be done before graph finalization
         *
         * @return The graph configuration
         */
        const BIGraphConfig &config() const;
        /** Sets graph configuration
         *
         * @param[in] config Configuration to use
         */
        void set_config(const BIGraphConfig &config);
        /** Inserts a memory manager context
         *
         * @param[in] memory_ctx Memory manage context
         *
         * @return True if the insertion succeeded else false
         */
        bool insert_memory_management_ctx(BIMemoryManagerContext &&memory_ctx);
        /** Gets a memory manager context for a given target
         *
         * @param[in] target To retrieve the management context
         *
         * @return Management context for the target if exists else nullptr
         */
        BIMemoryManagerContext *memory_management_ctx(BITarget target);
        /** Gets the memory managers map
         *
         * @return Memory manager contexts
         */
        std::map<BITarget, BIMemoryManagerContext> &memory_managers();
        /** Inserts a weights manager context
         *
         * @param[in] weights_ctx Weights manager context
         *
         * @return True if the insertion succeeded else false
         */
        bool insert_weights_management_ctx(BIWeightsManagerContext &&weights_ctx);

        /** Gets a weights manager context for a given target
         *
         * @param[in] target To retrieve the weights management context
         *
         * @return Management context for the target if exists else nullptr
         */
        BIWeightsManagerContext *weights_management_ctx(BITarget target);

        /** Gets the weights managers map
         *
         * @return Weights manager contexts
         */
        std::map<BITarget, BIWeightsManagerContext> &weights_managers();
        /** Finalizes memory managers in graph context */
        void finalize();

    private:
        BIGraphConfig                               _config;           /**< Graph configuration */
        std::map<BITarget, BIMemoryManagerContext>  _memory_managers;  /**< Memory managers for each target */
        std::map<BITarget, BIWeightsManagerContext> _weights_managers; /**< Weights managers for each target */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_GRAPHCONTEXT_H
