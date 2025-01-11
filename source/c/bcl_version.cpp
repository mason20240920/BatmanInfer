//
// Created by Mason on 2025/1/10.
//

#include "bcl_version.hpp"

namespace {
    constexpr BclVersion version_info{
            BI_COMPUTE_LIBRARY_VERSION_MAJOR,
            BI_COMPUTE_LIBRARY_VERSION_MINOR,
            BI_COMPUTE_LIBRARY_VERSION_PATCH,
    };
    extern "C" const BclVersion *BclVersionInfo() {
        return &version_info;
    }
}