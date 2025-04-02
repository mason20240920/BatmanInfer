//
// Created by Mason on 2025/4/1.
//

#include <runtime/utils.hpp>

#include <runtime/neon/bi_ne_scheduler.hpp>
#include <map>
#include <string>

namespace BatmanInfer {
    namespace utils {
#ifndef DOXYGEN_SKIP_THIS
        static const std::string information = "BatmanInfer";
#endif /* DOXYGEN_SKIP_THIS */

        const std::string &string_from_scheduler_type(BIScheduler::Type t) {
            static std::map<BIScheduler::Type, const std::string> scheduler_type_map = {
                {BIScheduler::Type::ST, "Single Thread"},
                {BIScheduler::Type::CPP, "C++11 Threads"},
                {BIScheduler::Type::OMP, "OpenMP Threads"},
                {BIScheduler::Type::CUSTOM, "Custom"}
            };

            return scheduler_type_map[t];
        }

        void schedule_kernel_on_ctx(BIIRuntimeContext *ctx, BIICPPKernel *kernel, const BIIScheduler::Hints &hints) {
            if (ctx) {
                BI_COMPUTE_ERROR_ON(ctx->scheduler() == nullptr);
                ctx->scheduler()->schedule(kernel, hints);
            } else {
                BINEScheduler::get().schedule(kernel, hints);
            }
        }

        unsigned int calculate_number_of_stages_only_x_axis(size_t input_x_dimension, unsigned int axis) {
            // We need only 1 stage for all axis except x-axis
            if (axis != 0) {
                return 1;
            }
            // Calculate number of WGs. 16 elements per thread, 8 threads per WG
            const auto num_of_wg = static_cast<unsigned int>(ceil(input_x_dimension / 128.f));

            // Calculate number of stages. First stage performs op and the rest reduction sum
            // depending on the size of the input. Last stage should have only 1 WG.
            const unsigned int num_of_stages = num_of_wg / 128 + 2;
            return num_of_stages;
        }
    } // namespace utils
}
