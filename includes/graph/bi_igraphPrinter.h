//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_IGRAPHPRINTER_H
#define BATMANINFER_GRAPH_BI_IGRAPHPRINTER_H

#include <ostream>

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;

    /** Graph printer interface */
    class BIIGraphPrinter
    {
    public:
        /** Virtual Destructor */
        virtual ~BIIGraphPrinter() = default;
        /** Print graph
         *
         * @param[in]  g  Graph to print
         * @param[out] os Output stream
         */
        virtual void print(const BIGraph &g, std::ostream &os) = 0;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_IGRAPHPRINTER_H
