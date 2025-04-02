//
// Created by Mason on 2025/4/1.
//

#pragma once

#include <runtime/bi_i_runtime_context.hpp>
#include <runtime/bi_scheduler.hpp>

#include <string>

namespace BatmanInfer {
    namespace utils {
        /** Convert a Scheduler::Type into a string.
         *
         * @param[in] t @ref Scheduler::Type to be translated to string.
         *
         * @return The string describing the scheduler type.
         */
        const std::string &string_from_scheduler_type(BIScheduler::Type t);

        /** Schedules a kernel using the context if not nullptr else uses the legacy scheduling flow.
         *
         * @param[in] ctx    Context to use.
         * @param[in] kernel Kernel to schedule.
         * @param[in] hints  Hints to use.
         */
        void schedule_kernel_on_ctx(BIIRuntimeContext *ctx, BIICPPKernel *kernel,
                                    const BIIScheduler::Hints &hints);

        /** Calculate number of stages for parallel implementations
         *
         * @param[in] input_x_dimension input tensor x dimension
         * @param[in] axis              axis to be used
         */
        unsigned int calculate_number_of_stages_only_x_axis(size_t input_x_dimension, unsigned int axis);
    } // namespace utils
}
