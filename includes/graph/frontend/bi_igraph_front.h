//
// Created by holynova on 2025/2/4.
//

#pragma once

#include "graph/frontend/bi_types.h"

namespace BatmanInfer {

namespace graph {

// Forward declarations
class BIGraph;

namespace frontend {

    // Forward declarations
    class BIILayer;

    /** graph frontend interface **/
    class BIIGraphFront {
    public:
        virtual ~BIIGraphFront() = default;

        /** Adds a layer to the frontend
         *
         * @param[in] layer Layer to add
         */
        virtual void add_layer(BIILayer &layer) = 0;

        /** Returns the underlying graph
         *
         * @return Underlying graph
         */
        virtual BIGraph &graph() = 0;

        /** Returns the underlying graph
         *
         * @return Underlying graph
         */
        virtual const BIGraph &graph() const = 0;

        /** Returns the tail node of the frontend
         *
         * @return Tail Node ID
         */
        NodeID tail_node()
        {
            return _tail_node;
        }

        /** Returns the hints that are currently used
         *
         * @return Frontend hints
         */
        FrontendHints &hints()
        {
            return _hints;
        }

        /** Forwards tail of frontend to a given nid
         *
         * @param[in] nid NodeID of the updated tail node
         */
        void forward_tail(NodeID nid)
        {
            _tail_node = (nid != NullTensorID) ? nid : _tail_node;
        }

        void set_target(const BITarget target_hint) {
            _hints.target_hint = target_hint;
        }

        void set_fast_math_hit(const BIFastMathHint fast_math_hint) {
            _hints.fast_math_hint = fast_math_hint;
        }

        void add_layer_rhs(BIILayer &&layer) {
            add_layer(layer);
        }

    protected:
        FrontendHints _hints     = {};
        NodeID        _tail_node = { EmptyNodeID };
    };

} // namespace frontend

} // namespace graph

} // namespace BatmanInfer
