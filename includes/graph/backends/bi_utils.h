//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_UTILS_H
#define BATMANINFER_GRAPH_BI_UTILS_H

#include "graph/bi_graphContext.h"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_i_weights_manager.hpp"
#include "runtime/bi_i_function.hpp"

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** Creates and configures a named function
     *
     * @param[in] name Name of the function
     * @param[in] args Function arguments
     *
     * @return  A configured backend function
     */
    template <typename FunctionType, typename FunctionNameType, typename... ParameterType>
    std::tuple<std::unique_ptr<BatmanInfer::BIIFunction>, FunctionNameType> create_named_function(FunctionNameType name,
                                                                                                  ParameterType... args)
    {
        auto f = std::make_unique<FunctionType>();
        f->configure(std::forward<ParameterType>(args)...);
        return std::make_pair(std::move(f), name);
    }

    /** Creates and configures a named function
     *
     * @param[in] name Name of the function
     * @param[in] mm   Memory manager to use
     * @param[in] args Function arguments
     *
     * @return  A configured backend function
     */
    template <typename FunctionType, typename FunctionNameType, typename MemoryManagerType, typename... ParameterType>
    std::tuple<std::unique_ptr<BatmanInfer::BIIFunction>, FunctionNameType>
    create_named_memory_managed_function(FunctionNameType name, MemoryManagerType mm, ParameterType... args)
    {
        auto f = std::make_unique<FunctionType>(mm);
        f->configure(std::forward<ParameterType>(args)...);
        return std::make_pair(std::move(f), name);
    }

    /** Checks if an operation is in place
     *
     * @param[in] input  Pointer to input
     * @param[in] output Pointer to output
     *
     * @return True if output is nullptr or input is equal to the output, else false
     */
    inline bool is_in_place_operation(void *input, void *output)
    {
        return (output == nullptr) || (input == output);
    }

    /** Returns the memory manager for a given target
     *
     * @param[in] ctx    Graph context containing memory management metadata
     * @param[in] target Target to retrieve the memory manager from
     *
     * @return The memory manager for the given target else false
     */
    inline std::shared_ptr<BIIMemoryManager> get_memory_manager(BIGraphContext &ctx, BITarget target)
    {
        bool enabled = ctx.config().use_function_memory_manager && (ctx.memory_management_ctx(target) != nullptr);
        return enabled ? ctx.memory_management_ctx(target)->intra_mm : nullptr;
    }

    /** Returns the weights manager for a given target
     *
     * @param[in] ctx    Graph context containing weight management metadata
     * @param[in] target Target to retrieve the weights manager from
     *
     * @return The weights manager for the given target else false
     */
    inline std::shared_ptr<BIIWeightsManager> get_weights_manager(BIGraphContext &ctx, BITarget target)
    {
        bool enabled = ctx.config().use_function_weights_manager && (ctx.weights_management_ctx(target) != nullptr);
        return enabled ? ctx.weights_management_ctx(target)->wm : nullptr;
    }

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_UTILS_H
