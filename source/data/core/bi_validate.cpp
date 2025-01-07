//
// Created by Mason on 2025/1/7.
//

#include <data/core/bi_vlidate.hpp>

BatmanInfer::BIStatus BatmanInfer::error_on_unconfigured_kernel(const char *function, const char *file, const int line,
                                                                const BatmanInfer::BIIKernel *kernel) {
    BI_COMPUTE_RETURN_ERROR_ON_LOC(kernel == nullptr, function, file, line);
    BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(!kernel->is_window_configured(), function, file, line,
                                       "This kernel hasn't been configured.");
    return BatmanInfer::BIStatus{};
}

BatmanInfer::BIStatus BatmanInfer::error_on_invalid_subwindow(const char *function, const char *file, const int line,
                                                              const BatmanInfer::BIWindow &full,
                                                              const BatmanInfer::BIWindow &sub) {
    full.validate();
    sub.validate();

    for (size_t i = 0; i < BatmanInfer::BICoordinates::num_max_dimensions; ++i) {
        BI_COMPUTE_RETURN_ERROR_ON_LOC(full[i].start() > sub[i].start(), function, file, line);
        BI_COMPUTE_RETURN_ERROR_ON_LOC(full[i].end() < sub[i].end(), function, file, line);
        BI_COMPUTE_RETURN_ERROR_ON_LOC(full[i].step() != sub[i].step(), function, file, line);
        BI_COMPUTE_RETURN_ERROR_ON_LOC((sub[i].start() - full[i].start()) % sub[i].step(), function, file, line);
    }
    return BatmanInfer::BIStatus{};
}