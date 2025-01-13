//
// Created by holynova on 2025/1/13.
//

#ifndef BATMANINFER_GRAPH_BI_IGRAPHMUTATOR_H
#define BATMANINFER_GRAPH_BI_IGRAPHMUTATOR_H

namespace BatmanInfer {

    namespace graph {

        // Forward declarations
        class BIGraph;

        /** Graph mutator interface */
        class BIIGraphMutator
        {
        public:
            /** Mutation type */
            enum class MutationType
            {
                IR,     /** IR specific mutation */
                Backend /** Backend specific mutation */
            };

        public:
            /** Virtual Destructor */
            virtual ~BIIGraphMutator() = default;
            /** Walk the graph and perform a specific mutation
             *
             * @param[in, out] g Graph to walk and mutate
             */
            virtual void mutate(BIGraph &g) = 0;
            /** Returns mutation type
             *
             * @return Mutation type enumeration
             */
            virtual MutationType type() const = 0;
            /** Returns mutator name
             *
             * @return Mutator name
             */
            virtual const char *name() = 0;
        };

    } // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_IGRAPHMUTATOR_H
