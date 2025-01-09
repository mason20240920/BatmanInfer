//
// Created by Mason on 2025/1/9.
//

#include <cpu/kernels/activation/heuristics/bi_cpu_activation_kernel_heuristics.hpp>

#include <data/core/utils/data_type_utils.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/bi_registers.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
//#include <cpu/kernels/activation/li