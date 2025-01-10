//
// Created by holynova on 2025/1/10.
//

#include "graph/backends/bi_backendRegistry.h"

using namespace BatmanInfer::graph::backends;

namespace BatmanInfer {

namespace graph {

namespace backends {

    BIBackendRegistry::BIBackendRegistry() : _registered_backends()
    {}

    BIBackendRegistry &BIBackendRegistry::get()
    {
        static BIBackendRegistry instance;
        return instance;
    }

    BIIDeviceBackend *BIBackendRegistry::find_backend(BITarget target)
    {
        BI_COMPUTE_ERROR_ON(!contains(target));
        return _registered_backends[target].get();
    }

    BIIDeviceBackend &BIBackendRegistry::get_backend(BITarget target)
    {
        BIIDeviceBackend *backend = find_backend(target);
        BI_COMPUTE_ERROR_ON_MSG(!backend, "Requested Backend doesn't exist!");
        BI_COMPUTE_ERROR_ON_MSG(!backend->is_backend_supported(), "Requested Backend is not supported!");
        return *backend;
    }

    bool BIBackendRegistry::contains(BITarget target) const
    {
        auto it = _registered_backends.find(target);
        return (it != _registered_backends.end());
    }

    const std::map<BITarget, std::unique_ptr<BIIDeviceBackend>> &BIBackendRegistry::backends() const
    {
        return _registered_backends;
    }

} // namespace backends

} // namespace graph

} // namespace BatmanInfer



