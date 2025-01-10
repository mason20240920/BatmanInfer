//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_NENODEVALIDATOR_H
#define BATMANINFER_GRAPH_BI_NENODEVALIDATOR_H

#include "data/core/bi_error.h"

namespace BatmanInfer {

namespace graph {

// Forward declarations
class BIINode;

namespace backends {

    class BINENodeValidator final
    {
    public:
        /** Validate a node
         *
         * @param[in] node Node to validate
         *
         * @return An error status
         */
        static BIStatus validate(BIINode *node);
    };

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_NENODEVALIDATOR_H
