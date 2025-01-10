//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_NEFUNCTIONFACTORY_H
#define BATMANINFER_GRAPH_BI_NEFUNCTIONFACTORY_H

#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

namespace graph {

// Forward declarations
class BIINode;
class BIGraphContext;

namespace backends {

    /** Factory for generating CPU backend functions **/
    class BINEFunctionFactory final
    {
    public:
        /** Create a backend execution function depending on the node type
         *
         * @param[in] node Node to create the backend function for
         * @param[in] ctx  Context to use
         *
         * @return Backend function
         */
        static std::unique_ptr<BatmanInfer::BIIFunction> create(BIINode *node, BIGraphContext &ctx);
    };

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_NEFUNCTIONFACTORY_H
