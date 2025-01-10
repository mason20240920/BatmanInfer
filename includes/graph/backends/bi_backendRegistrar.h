//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_BACKENDREGISTRAR_H
#define BATMANINFER_GRAPH_BI_BACKENDREGISTRAR_H

#include "graph/backends/bi_backendRegistry.h"

#include <utility>

namespace BatmanInfer {

namespace graph {

namespace backends {

namespace detail {

    /** Helper class to statically register a backend */
    template <typename T>
    class BackendRegistrar final
    {
    public:
        /** Add a new backend to the backend registry
         *
         * @param[in] target Execution target
         */
        BackendRegistrar(BITarget target);
    };

    template <typename T>
    inline BackendRegistrar<T>::BackendRegistrar(BITarget target)
    {
        BIBackendRegistry::get().add_backend<T>(target);
    }

} // namespace detail

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_BACKENDREGISTRAR_H
