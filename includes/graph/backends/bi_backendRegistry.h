//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_BACKENDREGISTRY_H
#define BATMANINFER_GRAPH_BI_BACKENDREGISTRY_H

#include "graph/bi_ideviceBackend.h"
#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** Registry holding all the supported backends */
    class BIBackendRegistry final
    {
    public:
        /** Gets backend registry instance
         *
         * @return Backend registry instance
         */
        static BIBackendRegistry &get();
        /** Finds a backend in the registry
         *
         * @param[in] target Backend target
         *
         * @return Pointer to the backend interface if found, else nullptr
         */
        BIIDeviceBackend *find_backend(BITarget target);

        /** Get a backend from the registry
         *
         * The backend must be present and supported.
         *
         * @param[in] target Backend target
         *
         * @return Reference to the backend interface
         */
        BIIDeviceBackend &get_backend(BITarget target);
        /** Checks if a backend for a given target exists
         *
         * @param[in] target Execution target
         *
         * @return True if exists else false
         */
        bool contains(BITarget target) const;
        /** Backends accessor
         *
         * @return Map containing the registered backends
         */
        const std::map<BITarget, std::unique_ptr<BIIDeviceBackend>> &backends() const;
        /** Registers a backend to the registry
         *
         * @param[in] target Execution target to register for
         */
        template <typename T>
        void add_backend(BITarget target);

    private:
        /** Default Constructor */
        BIBackendRegistry();

    private:
        std::map<BITarget, std::unique_ptr<BIIDeviceBackend>> _registered_backends;
    };

    template <typename T>
    inline void BIBackendRegistry::add_backend(BITarget target)
    {
        _registered_backends[target] = std::make_unique<T>();
    }

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_BACKENDREGISTRY_H
