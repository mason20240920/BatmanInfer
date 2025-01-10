//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_COMMON_TYPES_HPP
#define BATMANINFER_BI_COMMON_TYPES_HPP

#include <bcl_descriptors.hpp>
#include <bcl_types.hpp>

namespace BatmanInfer {
    enum class StatusCode {
        Success = BclSuccess,
        RuntimeError = BclRuntimeError,
        OutOfMemory = BclOutOfMemory,
        Unimplemented = BclUnimplemented,
        UnsupportedTarget = BclUnsupportedTarget,
        InvalidTarget = BclInvalidTarget,
        InvalidArgument = BclInvalidArgument,
        UnsupportedConfig = BclUnsupportedConfig,
        InvalidObjectState = BclInvalidObjectState,
    };

    enum class Target {
        Cpu = BclTarget::BclCpu,
        GpuOcl = BclTarget::BclGpuOcl,
    };

    enum class ExecutionMode {
        FastRerun = BclPreferFastRerun,
        FastStart = BclPreferFastStart,
    };

    enum class ImportMemoryType {
        HostPtr = BclImportMemoryType::BclHostPtr
    };
}

#endif //BATMANINFER_BI_COMMON_TYPES_HPP
