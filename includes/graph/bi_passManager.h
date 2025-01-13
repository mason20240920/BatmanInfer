//
// Created by holynova on 2025/1/13.
//

#ifndef BATMANINFER_GRAPH_BI_PASSMANAGER_H
#define BATMANINFER_GRAPH_BI_PASSMANAGER_H

#include "graph/bi_igraphMutator.h"

#include <memory>
#include <vector>

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;

    /** Pass manager
     *
     * Responsible for performing the mutating graph passes with a given order
     **/
    class BIPassManager final
    {
    public:
        /** Constructor */
        BIPassManager();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIPassManager(const BIPassManager &) = delete;
        /** Default move constructor */
        BIPassManager(BIPassManager &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIPassManager &operator=(const BIPassManager &) = delete;
        /** Default move assignment operator */
        BIPassManager &operator=(BIPassManager &&) = default;
        /** Mutation passes accessors
         *
         * @return Returns the vector with the mutation passes that are to be executed on a graph
         */
        const std::vector<std::unique_ptr<BIIGraphMutator>> &passes() const;
        /** Accessor of a pass at a given index
         *
         * @param[in] index Index of the requested pass
         *
         * @return A pointer to the given pass if exists else nullptr
         */
        BIIGraphMutator *pass(size_t index);
        /** Appends a mutation pass
         *
         * @param[in] pass        Pass to append
         * @param[in] conditional (Optional) Append pass if true else false. Defaults to true.
         */
        void append(std::unique_ptr<BIIGraphMutator> pass, bool conditional = true);
        /** Clears all the passes */
        void clear();
        /** Runs all the mutation passes on a given graph
         *
         * @param[in, out] g Graph to run the mutations on
         */
        void run_all(BIGraph &g);
        /** Runs a mutation passes of a specific type on a given graph
         *
         * @param[in, out] g    Graph to run the mutation on
         * @param[in]      type Type of the mutations to execute
         */
        void run_type(BIGraph &g, BIIGraphMutator::MutationType type);
        /** Runs a specific mutation pass on a given graph
         *
         * @param[in, out] g     Graph to run the mutation on
         * @param[in]      index Index of the mutation to execute
         */
        void run_index(BIGraph &g, size_t index);

    private:
        std::vector<std::unique_ptr<BIIGraphMutator>> _passes; /**< Vector of graph passes */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_PASSMANAGER_H
