//
// Created by Mason on 2025/1/4.
//

#include <runtime/bi_single_thread_scheduler.hpp>

#include <data/core/cpp/bi_i_cpp_kernel.hpp>
#include <data/core/bi_error.h>
#include <data/core/bi_utils.hpp>

namespace BatmanInfer {
    void BISingleThreadScheduler::set_num_threads(unsigned int num_threads) {
        BI_COMPUTE_UNUSED(num_threads);
        BI_COMPUTE_ERROR_ON(num_threads != 1);
    }

    void BISingleThreadScheduler::schedule(BatmanInfer::BIICPPKernel *kernel,
                                           const BatmanInfer::BIIScheduler::Hints &hints) {
        const BIWindow &max_window = kernel->window();

        if (hints.split_dimension() != BIIScheduler::split_dimensions_all) {
            const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
            if (num_iterations < 1)
                return;
        }

        ThreadInfo info;
        info.cpu_info = &cpu_info();
        kernel->run(kernel->window(), info);
    }

    void BISingleThreadScheduler::schedule_op(BatmanInfer::BIICPPKernel *kernel,
                                              const BatmanInfer::BIIScheduler::Hints &hints,
                                              const BatmanInfer::BIWindow &window,
                                              BatmanInfer::BIITensorPack &tensors) {
        BI_COMPUTE_UNUSED(hints);
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        kernel->run_op(tensors, window, info);
    }

    void BISingleThreadScheduler::run_workloads(std::vector<BIWorkload> &workloads) {
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        for (auto &wl: workloads) {
            wl(info);
        }
    }

    unsigned int BISingleThreadScheduler::num_threads() const {
        return 1;
    }

}