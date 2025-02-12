//
// Created by holynova on 2025/2/4.
//

#pragma once

#include "graph/bi_types.h"

#include <string>

namespace BatmanInfer {

namespace graph {

namespace frontend {

    // Forward declarations
    class BIIGraphFront;

    /** ILayer interface */
    class BIILayer
    {
    public:
        /** Default destructor */
        virtual ~BIILayer() = default;

        /** Create layer and add to the given graph frontend.
         *
         * @param[in] gf Graph fontend to add layer to.
         *
         * @return ID of the created node.
         */
        virtual NodeID create_layer(BIIGraphFront &gf) = 0;

        /** Sets the name of the layer
         *
         * @param[in] name Name of the layer
         *
         * @return The layer object
         */
        BIILayer &set_name(std::string name)
        {
            _name = name;
            return *this;
        }

        /** Layer name accessor
         *
         * @return Returns the name of the layer
         */

        const std::string &name() const
        {
            return _name;
        }

    private:
        std::string _name = {};
    };

} // namespace frontend

} // namespace graph

} // namespace BatmanInfer
